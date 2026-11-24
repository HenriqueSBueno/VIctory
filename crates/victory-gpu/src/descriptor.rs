//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (Grand Theft Auto VI) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// Bytes per descriptor in this heap.
    #[cfg(windows)]
    pub(crate) fn step(&self) -> u32 {
        increment(self.desc.kind).unwrap_or(0)
    }

    /// Total descriptor slots in this title heap.
    #[cfg(windows)]
    pub(crate) fn count(&self) -> u32 {
        self.desc.count
    }

    /// Total bytes of descriptor heap backing memory.
    #[cfg(windows)]
    pub(crate) fn bytes(&self) -> u64 {
        u64::from(self.step()) * u64::from(self.desc.count.max(1))
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of Grand Theft Auto VI's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
static SHADER_HEAPS: std::sync::Mutex<Vec<(u64, u64, usize)>> = std::sync::Mutex::new(Vec::new());

/// The shader-visible heap holding `handle`, and the descriptor's index.
#[cfg(windows)]
fn shader_slot<'a>(handle: u64) -> Option<(&'a Heap, u32)> {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let &(_, _, at) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))?;
    // SAFETY: registered heaps are live (removed when dropped).
    let heap = unsafe { &*(at as *const Heap) };
    Some((heap, heap.index(handle)?))
}

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap.
#[cfg(windows)]
fn mirror(handle: u64) {
    let Some((heap, index)) = shader_slot(handle) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// The descriptor record `[tag, index, generation]` at `handle`, if it
/// holds a view of ours.
#[cfg(windows)]
pub(crate) fn slot_record(handle: u64) -> Option<[u32; 3]> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    (record[0] == VIEW_TAG && record[2] != 0).then_some(record)
}

/// Maximum number of distinct first-dword patterns logged for unrecognised
/// 32-byte descriptor table slots in diagnostics builds.
#[cfg(any(feature = "diagnostics", test))]
pub(crate) const UNKNOWN_SLOT_LOG_LIMIT: usize = 64;

/// Classification of the 8 little-endian dwords (32 bytes) in a view
/// descriptor heap slot.
#[cfg(any(feature = "diagnostics", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlotRecord {
    /// All 32 bytes are zero.
    Zero,
    /// VIctory's own `[VIEW_TAG, index, generation]` record (`generation != 0`).
    Victory { index: u32, generation: u32 },
    /// Non-zero bytes that are not a recognised VIctory descriptor record.
    Unrecognised([u32; 8]),
}

/// Classifies the 8 dwords of a 32-byte view descriptor slot in title memory.
#[cfg(any(feature = "diagnostics", test))]
pub(crate) fn classify_slot(dwords: [u32; 8]) -> SlotRecord {
    if dwords == [0; 8] {
        SlotRecord::Zero
    } else if dwords[0] == VIEW_TAG && dwords[2] != 0 {
        SlotRecord::Victory {
            index: dwords[1],
            generation: dwords[2],
        }
    } else {
        SlotRecord::Unrecognised(dwords)
    }
}

/// Formats an 8-dword descriptor slot as zero-padded hex words.
#[cfg(any(feature = "diagnostics", test))]
pub(crate) fn format_dwords(dwords: &[u32; 8]) -> String {
    format!(
        "[{:#010x}, {:#010x}, {:#010x}, {:#010x}, {:#010x}, {:#010x}, {:#010x}, {:#010x}]",
        dwords[0], dwords[1], dwords[2], dwords[3], dwords[4], dwords[5], dwords[6], dwords[7]
    )
}

/// Bounded deduplicating filter for unrecognised descriptor slot patterns,
/// keyed by the first dword (`dwords[0]`).
#[cfg(any(feature = "diagnostics", test))]
pub(crate) struct UnknownSlotLog {
    seen: Vec<u32>,
    limit: usize,
}

#[cfg(any(feature = "diagnostics", test))]
impl UnknownSlotLog {
    pub(crate) const fn new(limit: usize) -> Self {
        Self {
            seen: Vec::new(),
            limit,
        }
    }

    /// Returns `true` the first time `first_dword` is seen while under the
    /// cap.
    pub(crate) fn should_log(&mut self, first_dword: u32) -> bool {
        if self.seen.len() >= self.limit || self.seen.contains(&first_dword) {
            return false;
        }
        self.seen.push(first_dword);
        true
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.seen.len()
    }
}

#[cfg(feature = "diagnostics")]
static UNKNOWN_SLOTS: std::sync::Mutex<UnknownSlotLog> =
    std::sync::Mutex::new(UnknownSlotLog::new(UNKNOWN_SLOT_LOG_LIMIT));

#[cfg(feature = "diagnostics")]
static UNKNOWN_SLOT_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Logs an unrecognised non-zero 32-byte table slot once per distinct
/// first-dword pattern (up to [`UNKNOWN_SLOT_LOG_LIMIT`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn note_unknown_dwords(
    heap_index: u32,
    dwords: [u32; 8],
    root_param: u8,
    user_data: Option<u32>,
) {
    use std::sync::atomic::Ordering;
    if UNKNOWN_SLOT_COUNT.load(Ordering::Relaxed) >= UNKNOWN_SLOT_LOG_LIMIT {
        return;
    }
    let SlotRecord::Unrecognised(dwords) = classify_slot(dwords) else {
        return;
    };
    let should = {
        let mut log = UNKNOWN_SLOTS.lock().unwrap_or_else(|e| e.into_inner());
        let ok = log.should_log(dwords[0]);
        if ok {
            UNKNOWN_SLOT_COUNT.store(log.seen.len(), Ordering::Relaxed);
        }
        ok
    };
    if should {
        tracing::warn!(
            heap_index,
            root_param,
            ?user_data,
            dwords = %format_dwords(&dwords),
            "unrecognised non-zero descriptor slot in table"
        );
    }
}

/// Reads 32 bytes from a live view descriptor heap slot at `handle` and logs it
/// if it is neither all zero nor a recognised VIctory descriptor record.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_unknown_handle(
    heap_index: u32,
    handle: u64,
    root_param: u8,
    user_data: Option<u32>,
) {
    if handle == 0
        || UNKNOWN_SLOT_COUNT.load(std::sync::atomic::Ordering::Relaxed) >= UNKNOWN_SLOT_LOG_LIMIT
    {
        return;
    }
    // SAFETY: callers pass a handle inside a live 32-byte view heap slot.
    let dwords = unsafe { (handle as *const [u32; 8]).read_unaligned() };
    note_unknown_dwords(heap_index, dwords, root_param, user_data);
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(first, n);
    }
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        if self.memory.is_some()
            && self.desc.kind == VIEWS_KIND
            && self.desc.flags & SHADER_VISIBLE != 0
        {
            crate::shadow::unregister(at, self.start);
        }
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // Render target and depth views: a packed 16-bit target may be
        // R8G8B8A8_UNORM on this GPU (`resource::target_format`).
        words[0] = crate::resource::target_format(words[0]);
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
/// shortcut: views without a resource object (constant buffers, memory
/// views, samplers) are kept for the run; they are bounded by the title's
/// memory and sampler states.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
}

static VIEWS: std::sync::Mutex<Views> = std::sync::Mutex::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match views.index.get(&view) {
        Some(&i) => i,
        None => {
            let i = match views.free.pop() {
                Some(i) => {
                    views.list[i as usize].0 = view.clone();
                    i
                }
                None => {
                    views.list.push((view.clone(), 1));
                    views.list.len() as u32 - 1
                }
            };
            for r in view.resources().into_iter().filter(|&r| r != 0) {
                views.by_resource.entry(r).or_default().push(i);
            }
            views.index.insert(view, i);
            i
        }
    };
    let generation = views.list[i as usize].1;
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    drop(views);
    #[cfg(windows)]
    mirror(handle);
}

#[cfg(windows)]
static FORGET_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    let mut forgot = false;
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        views.list[i as usize].1 = generation.wrapping_add(1).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
        forgot = true;
    }
    #[cfg(windows)]
    if forgot {
        FORGET_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Release);
    }
    #[cfg(not(windows))]
    let _ = forgot;
}

/// Monotonic epoch bumped whenever any view entry is invalidated by [`forget`].
#[cfg(windows)]
pub(crate) fn forget_epoch() -> u64 {
    FORGET_EPOCH.load(std::sync::atomic::Ordering::Acquire)
}

/// The view of entry `i` if `generation` is still its own, together with the
/// [`forget_epoch`] observed under the `VIEWS` lock.
#[cfg(windows)]
pub(crate) fn view_at_epoch(i: u32, generation: u32) -> Option<(View, u64)> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (*current == generation).then(|| {
        (
            view.clone(),
            FORGET_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
        )
    })
}

/// Batch liveness check for `(index, generation)` pairs under a single `VIEWS`
/// lock acquisition, returning the [`forget_epoch`] observed under the lock.
#[cfg(windows)]
pub(crate) fn check_live(pairs: &[(u32, u32)], out: &mut Vec<bool>) -> u64 {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    out.clear();
    out.extend(pairs.iter().map(|&(i, generation)| {
        views
            .list
            .get(i as usize)
            .is_some_and(|(_, current)| *current == generation)
    }));
    FORGET_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (tag == VIEW_TAG && *current == generation).then(|| view.clone())
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    }
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let step = u64::from(increment(kind).unwrap_or(0));
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let view_watched = !placed && visible && desc.kind == VIEWS_KIND;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        (placement.address, None)
    } else if view_watched {
        match Memory::zeroed_watched(size) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
    });
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        if view_watched {
            crate::shadow::register(at, start, size);
        }
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = (visible, view_watched);
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// Grand Theft Auto VI places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }

    #[test]
    fn classify_slot_distinguishes_zero_victory_and_unrecognised() {
        assert_eq!(classify_slot([0; 8]), SlotRecord::Zero);

        let victory = [VIEW_TAG, 7, 1, 0, 0, 0, 0, 0];
        assert_eq!(
            classify_slot(victory),
            SlotRecord::Victory {
                index: 7,
                generation: 1
            }
        );

        let victory_trailing = [VIEW_TAG, 7, 1, 0xdead_beef, 1, 2, 3, 4];
        assert_eq!(
            classify_slot(victory_trailing),
            SlotRecord::Victory {
                index: 7,
                generation: 1
            }
        );

        let zero_gen = [VIEW_TAG, 7, 0, 0, 0, 0, 0, 0];
        assert_eq!(classify_slot(zero_gen), SlotRecord::Unrecognised(zero_gen));

        let nonzero_tail = [0, 0x1000, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            classify_slot(nonzero_tail),
            SlotRecord::Unrecognised(nonzero_tail)
        );

        let raw_hw = [0x1234_5678, 1, 0x100, 0x9000_0fac, 0, 0, 0, 0];
        assert_eq!(classify_slot(raw_hw), SlotRecord::Unrecognised(raw_hw));
    }

    #[test]
    fn unknown_slot_log_deduplicates_first_dword_and_enforces_limit() {
        assert_eq!(UNKNOWN_SLOT_LOG_LIMIT, 64);
        let mut log = UnknownSlotLog::new(3);
        assert!(log.should_log(0x1111_0000));
        assert!(!log.should_log(0x1111_0000));
        assert!(log.should_log(0x2222_0000));
        assert!(log.should_log(0x3333_0000));
        assert!(!log.should_log(0x4444_0000));
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn format_dwords_formats_all_eight_words_in_hex() {
        let dwords = [0, 1, 0x1234_abcd, 0xffff_ffff, 0x10, 0x20, 0x30, 0x40];
        assert_eq!(
            format_dwords(&dwords),
            "[0x00000000, 0x00000001, 0x1234abcd, 0xffffffff, 0x00000010, 0x00000020, 0x00000030, 0x00000040]"
        );
    }

    #[cfg(windows)]
    #[test]
    fn non_placed_shader_visible_view_heap_registers_watched_shadow_and_reads_slot_dwords() {
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 128,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = std::ptr::null();
        assert_eq!(create(desc, none, &mut h), S_OK);
        let view_heap = heap(h);
        assert_eq!(view_heap.bytes(), 128 * 32);
        let start = view_heap.start;
        assert!(crate::shadow::is_registered(h as usize, start));
        let shadow = crate::shadow::find(start, view_heap.bytes()).expect("view heap shadow");
        assert_eq!(shadow.left(start), view_heap.bytes());

        let slot3_off = 3 * 32u64;
        let payload = [
            1.0f32.to_bits(),
            6500.0f32.to_bits(),
            0.25f32.to_bits(),
            0x3f80_0000u32,
        ];
        // SAFETY: slot 3 (offset 96..112) lies within the 128 * 32 byte view heap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                (start + slot3_off) as *mut u32,
                payload.len(),
            );
        }
        assert_eq!(view_heap.index(start + slot3_off), Some(3));
        assert_eq!(
            (start + slot3_off) - slot::<extern "system" fn(This) -> u64>(h, 0x50)(h),
            slot3_off
        );
        let Some((now, spans)) =
            crate::watch::writes(start as usize, (start + view_heap.bytes()) as usize, false)
        else {
            panic!("expected MEM_WRITE_WATCH dirty spans");
        };
        let page = (start as usize) / crate::watch::PAGE;
        assert!(
            spans.contains(&(page, now)),
            "expected page {page} at asking {now} in {spans:?}"
        );

        assert_eq!(slot::<extern "system" fn(This) -> u32>(h, 0x10)(h), 0);
        assert!(!crate::shadow::is_registered(h as usize, start));

        // Sampler heaps do not register a view-heap data shadow.
        let sampler_desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let mut sh = std::ptr::null();
        assert_eq!(create(sampler_desc, none, &mut sh), S_OK);
        let sampler_heap = heap(sh);
        assert_eq!(sampler_heap.bytes(), 64 * 16);
        assert!(!crate::shadow::is_registered(sh as usize, sampler_heap.start));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(sh, 0x10)(sh), 0);
    }
}
