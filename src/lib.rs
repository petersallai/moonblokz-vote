#![no_std]

//! # moonblokz-vote
//!
//! Standalone MoonBlokz vote engine — a per-node accumulated-vote registry
//! that applies FR37 forward accumulation. Story 3.3 adds FR38 next-eligible-
//! creator selection.
//!
//! **Leaf-crate discipline.** The only direct dependency is
//! `moonblokz-chain-types` (for `BlockView<'_>` / `TransactionView<'_>` /
//! `ComplexTransactionView<'_>`). **No** direct dependency on
//! `moonblokz-blockchain`, crypto, or radio. `moonblokz-chain-types`
//! mandates one Schnorr backend feature, so `moonblokz-crypto` is present
//! in the transitive tree.
//!
//! The crate is fully deterministic and holds no PRNG: FR38 creator
//! ordering is deterministic (descending vote, ascending node-id
//! tie-break), and no planned story consumes vote-side randomness.
//!
//! Vote-target scoring is upstream / out of scope per ADR-007. The vote
//! engine consumes the `vote: u32` field already present on each transaction
//! (populated by the transaction author).
//!
//! ## FR37 forward accumulation (Story 3.1)
//!
//! [`VoteEngine::apply_block`] applies the three FR37 steps in this order:
//!
//! 1. **Anti-capture interest** — every node's `accumulated_vote` gets
//!    `av += min(floor(av × vote_interest / vote_scale), vote_scale)`
//!    using checked `u32` arithmetic, including the creator before the later
//!    reset. `vote_scale` is also the per-block interest bump cap.
//! 2. **Vote credits** — every payload transaction's `vote` target node
//!    gains one `vote_scale` credit, using checked `u32` arithmetic.
//!    Zero-input UTXO carry-forward complex
//!    transactions contribute no credit (FR51 replay-safety).
//! 3. **Creator reset** — the block creator's `accumulated_vote` is zeroed.
//!    Runs **last** so a creator voting for themselves in their own block
//!    does not retain those votes.
//!
//! Deviation-block penalty: a deviation block is one whose `first_voted_node`
//! differs from its creator (FR45); the `first_voted_node`'s accumulated vote is
//! also reset (grace-period penalty per FR47, once per fallback cycle).
//!
//! ## Header snapshots (FR37, Story 6.2 D5 / DEC-156)
//!
//! A block's `consumed_votes` is its creator's accumulated vote after step 1
//! and **before** step 2, and a deviation block's
//! `consumed_votes_from_first_voted_node` is the first-voted node's value at the
//! same point. A block's own credits therefore never enter its value, while
//! step 3 still wipes the credits a block gives its own creator. Because the
//! first-voted node leads the creator order (FR38) and interest is monotone, a
//! deviation block is never valued above the normal block it replaces.
//! [`VoteEngine::check_header`] verifies the snapshots (I-09, FR9 Tier 3), and
//! [`VoteEngine::undo_block`] restores both nodes from them.

use core::cmp::Ordering;
use core::num::NonZeroU16;
use moonblokz_chain_types::BlockView;

/// Errors returned when a vote-state transition would violate the reversible
/// `u32` accumulated-vote contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteEngineError {
    /// Applying interest or a vote credit would exceed `u32::MAX`.
    AccumulatedVoteOverflow,
    /// Reversing a vote credit would subtract below zero.
    AccumulatedVoteUnderflow,
    /// The post-interest value is not reachable from the configured growth function.
    UnreachableInterestState,
    /// The engine holds no FR37 parameters yet, so no vote effect can be
    /// computed. A node has none until a chain configuration is loaded
    /// (`moonblokz-blockchain` parameterizes the engine from the FR56
    /// configuration handle); reading this error means a caller reached a vote
    /// effect before that happened.
    NotParameterized,
}

/// The FR37 parameters, held together because they are set together and are
/// meaningless apart: `cap_threshold` is derived from the other two.
///
/// Bundled into one `Option` rather than three fields so that "parameterized"
/// is a single state rather than an invariant spread over three values. The
/// `NonZeroU16` gives `Option<VoteParams>` a niche, so the wrapper is free.
#[derive(Clone, Copy)]
#[cfg_attr(any(test, feature = "test-oracle"), derive(PartialEq, Eq))]
struct VoteParams {
    vote_scale: NonZeroU16,
    vote_interest: u8,
    // Smallest `av` for which `floor(av * vote_interest / vote_scale) >= vote_scale`.
    // Above this point the interest bump is capped at `vote_scale`, so growth is linear.
    cap_threshold: u32,
}

/// Per-node accumulated-vote registry (FR37 forward accumulation).
///
/// The engine is constructed **unparameterized**: the accumulated-vote array is
/// meaningful on its own (all zero), while the FR37 parameters are chain
/// configuration the node may not hold yet. Every vote effect therefore
/// answers [`VoteEngineError::NotParameterized`] until [`Self::reset`] supplies
/// them; the queries need no parameters and stay infallible.
///
/// Const generic:
/// - `MAX_NODES`: workspace node-roster capacity (architecture §5 default: 1000).
#[cfg_attr(any(test, feature = "test-oracle"), derive(PartialEq, Eq))]
pub struct VoteEngine<const MAX_NODES: usize> {
    accumulated_vote: [u32; MAX_NODES],
    params: Option<VoteParams>,
}

impl<const MAX_NODES: usize> VoteEngine<MAX_NODES> {
    /// Constructs the engine in place, inside caller-provided storage, and
    /// hands back a `&mut` to the initialized value. This is the type's
    /// **only** constructor: there is no by-value `new()`, because `Self`
    /// carries a `MAX_NODES * 4`-byte array (~3.9 KB at the architecture §5
    /// default `MAX_NODES = 1000`) and no construction technique *inside* a
    /// function that returns `Self` by value can avoid a
    /// `size_of::<Self>()`-sized transient stack allocation somewhere.
    /// Writing straight into `slot` sidesteps that floor entirely — see
    /// `moonblokz_blockchain::api::Blockchain::init`'s doc comment for the
    /// full mechanism and the required embedded usage pattern (a task-local
    /// `MaybeUninit` kept alive across an `.await`).
    ///
    /// - `vote_scale` / `vote_interest` — chain-config parameters read from
    ///   the caller's `ChainConfigTrait` at engine-construction time.
    ///   `vote_scale` is non-zero because FR37 uses it as both the denominator
    ///   of the anti-capture interest rule and the maximum per-block interest
    ///   bump. Each transaction vote credit is also worth one `vote_scale`.
    ///
    /// The engine is fully deterministic and takes no PRNG seed — FR38
    /// creator ordering needs no randomness (architecture §2.3, 2026-07-04
    /// revision). If chain-config becomes dynamic in a later story, the
    /// engine can be re-parameterized through [`Self::reset`].
    ///
    /// This function is **safe**: a `&mut MaybeUninit<Self>` already
    /// guarantees a non-null, aligned, exclusively borrowed destination that
    /// is valid for writes, and the caller only receives the `&mut Self` once
    /// every field has been written — a panic before that point leaves `slot`
    /// uninitialized, which is a safe state (nothing is dropped). If `slot`
    /// already held an initialized value it is leaked rather than dropped;
    /// every field here is plain data, so that is a no-op.
    pub fn init(slot: &mut core::mem::MaybeUninit<Self>) -> &mut Self {
        let p = slot.as_mut_ptr();
        // SAFETY: `p` is derived from a live `&mut MaybeUninit<Self>`, so it is
        // non-null, aligned, valid for writes of `Self` and not aliased for the
        // duration of this call. Both fields are written below and neither is
        // read before its write.
        unsafe {
            // All-zero `u32` array: `write_bytes` (memset) is correct (no
            // representation ambiguity for a primitive integer) and never
            // materializes a `MAX_NODES * 4`-byte value anywhere, unlike a
            // bulk `.write([0u32; MAX_NODES])` would. `count` here is a
            // count of `u32` elements, not bytes.
            let accumulated_vote_ptr = (&raw mut (*p).accumulated_vote).cast::<u32>();
            accumulated_vote_ptr.write_bytes(0u8, MAX_NODES);

            // A real `None`, written as a value: `Option<VoteParams>` is
            // niche-encoded, but that layout is not something `unsafe` code may
            // assume, so the discriminant is never memset into place.
            (&raw mut (*p).params).write(None);
        }
        // SAFETY: every field of `Self` was written above.
        unsafe { slot.assume_init_mut() }
    }

    /// Parameterizes and empties a live engine in place (FR3 "not resumable
    /// — clean working set on re-entry"), and is the **only** way an engine
    /// becomes usable: until it runs, every vote effect answers
    /// [`VoteEngineError::NotParameterized`].
    ///
    /// No `MAX_NODES`-scaled temporary is materialized: the array is zeroed
    /// where it already lives.
    pub fn reset(&mut self, vote_scale: NonZeroU16, vote_interest: u8) {
        self.clear();
        self.params = Some(VoteParams {
            vote_scale,
            vote_interest,
            cap_threshold: Self::compute_cap_threshold(vote_scale, vote_interest),
        });
    }

    /// Empties the accumulated-vote state, leaving the FR37 parameters as they
    /// are (or absent, if they never arrived).
    ///
    /// This is the FR3/FR5 "clean working set" step for the case where the
    /// caller has no configuration to parameterize with — it must still be able
    /// to guarantee that no partial projection survives.
    pub fn clear(&mut self) {
        // `fill` lowers to a memset over the array in place; it never
        // materializes a `MAX_NODES`-scaled value the way an assignment from a
        // literal would.
        self.accumulated_vote.fill(0);
    }

    /// The FR37 parameters, or the refusal every vote effect returns without
    /// them.
    fn params(&self) -> Result<VoteParams, VoteEngineError> {
        self.params.ok_or(VoteEngineError::NotParameterized)
    }

    /// The by-value constructor, kept as the **executable specification** of
    /// [`Self::init`]: a plain struct literal, which the language forces to
    /// name every field, so adding a field to `VoteEngine` is a compile error
    /// here until the literal — and therefore the specification — is updated.
    /// The equivalence test then holds `init` to it field by field.
    ///
    /// Test-only (`cfg(test)` or the `test-oracle` feature): returning `Self`
    /// by value costs a `size_of::<Self>()`-sized transient, which is exactly
    /// what `init` exists to avoid on the embedded stack. Never a production
    /// code path.
    // `Default` is deliberately not implemented: it would put a by-value
    // constructor on the production surface, which is exactly what `init`
    // exists to keep off it.
    #[allow(clippy::new_without_default)]
    #[cfg(any(test, feature = "test-oracle"))]
    pub fn new() -> Self {
        Self {
            accumulated_vote: [0u32; MAX_NODES],
            params: None,
        }
    }

    /// Test-only stand-in that goes through the production path: runs
    /// [`Self::init`] into a local slot, parameterizes it, and moves the
    /// finished value out.
    #[cfg(test)]
    fn new_for_test(vote_scale: NonZeroU16, vote_interest: u8) -> Self {
        let mut slot = core::mem::MaybeUninit::<Self>::uninit();
        let engine = Self::init(&mut slot);
        engine.reset(vote_scale, vote_interest);
        // SAFETY: `init` returned, so every field of `slot` is initialized.
        unsafe { slot.assume_init() }
    }

    fn compute_cap_threshold(vote_scale: NonZeroU16, vote_interest: u8) -> u32 {
        if vote_interest == 0 {
            return u32::MAX;
        }
        let scale = vote_scale.get() as u32;
        let interest = vote_interest as u32;
        // `u16::MAX * u16::MAX + u8::MAX` still fits in `u32`.
        (scale * scale).div_ceil(interest)
    }

    fn interest_bump_for(p: VoteParams, av: u32) -> u32 {
        if av == 0 || p.vote_interest == 0 {
            return 0;
        }
        let scale = p.vote_scale.get() as u32;
        if av >= p.cap_threshold {
            return scale;
        }
        // Safe below `cap_threshold`: `av * vote_interest < vote_scale^2`,
        // and `u16::MAX^2` fits in `u32`.
        (av * p.vote_interest as u32) / scale
    }

    fn apply_growth_to_value(p: VoteParams, av: u32) -> Result<u32, VoteEngineError> {
        av.checked_add(Self::interest_bump_for(p, av))
            .ok_or(VoteEngineError::AccumulatedVoteOverflow)
    }

    fn undo_growth_value(p: VoteParams, after: u32) -> Result<u32, VoteEngineError> {
        let scale = p.vote_scale.get() as u32;

        // Fast path for the capped linear region: `after = before + vote_scale`.
        if let Some(min_capped_after) = p.cap_threshold.checked_add(scale)
            && after >= min_capped_after
        {
            let before = after - scale;
            if before >= p.cap_threshold && Self::apply_growth_to_value(p, before) == Ok(after) {
                return Ok(before);
            }
        }

        // General exact inverse. `f(x) = x + bump(x)` is strictly increasing,
        // so a reachable `after` has exactly one pre-image. Any pre-image
        // satisfies `x = after - bump(x)`; with `bump` non-decreasing and
        // `x <= after`, `bump(x)` is pinned between `bump(lo)` and
        // `bump(after)`, bounding the pre-image to
        // `after - bump(after) <= x <= after - bump(lo)`. This shrinks the
        // search from `0..=after` to on the order of `vote_interest`
        // candidates. Saturating: degenerate configs
        // (`vote_interest > vote_scale`) can make `bump(after)` exceed a
        // small `after`.
        let mut lo = after.saturating_sub(Self::interest_bump_for(p, after));
        let mut hi = after.saturating_sub(Self::interest_bump_for(p, lo));
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            match Self::apply_growth_to_value(p, mid) {
                Ok(grown) => match grown.cmp(&after) {
                    Ordering::Equal => return Ok(mid),
                    Ordering::Less => {
                        if mid == u32::MAX {
                            break;
                        }
                        lo = mid + 1;
                    }
                    Ordering::Greater => {
                        if mid == 0 {
                            break;
                        }
                        hi = mid - 1;
                    }
                },
                Err(VoteEngineError::AccumulatedVoteOverflow) => {
                    if mid == 0 {
                        break;
                    }
                    hi = mid - 1;
                }
                Err(err) => return Err(err),
            }
        }

        Err(VoteEngineError::UnreachableInterestState)
    }

    /// Applies the FR37 three-step forward accumulation for `block`.
    ///
    /// See module-level docs for the step ordering and rationale. Returns an
    /// error instead of saturating if an accepted block would overflow the
    /// reversible `u32` accumulated-vote state.
    pub fn apply_block(&mut self, block: BlockView<'_>) -> Result<(), VoteEngineError> {
        let params = self.params()?;
        let creator = block.creator();
        let vote_scale = params.vote_scale.get() as u32;

        // Preflight all interest and vote-credit additions before mutating so
        // arithmetic errors fail closed and do not leave partially-updated state.
        for node_id in 0..MAX_NODES {
            let post_interest =
                Self::apply_growth_to_value(params, self.accumulated_vote[node_id])?;
            if let Some(payload) = block.transactions() {
                let mut credit_count = 0u32;
                for tx in payload.iter() {
                    if let Some(complex) = tx.as_complex()
                        && complex.input_count() == 0
                    {
                        continue;
                    }
                    if tx.vote() as usize == node_id {
                        credit_count = credit_count
                            .checked_add(1)
                            .ok_or(VoteEngineError::AccumulatedVoteOverflow)?;
                    }
                }
                if credit_count > 0 {
                    let credit_total = credit_count
                        .checked_mul(vote_scale)
                        .ok_or(VoteEngineError::AccumulatedVoteOverflow)?;
                    post_interest
                        .checked_add(credit_total)
                        .ok_or(VoteEngineError::AccumulatedVoteOverflow)?;
                }
            }
        }

        // === Step 1: anti-capture interest to every node, including creator ===
        for node_id in 0..MAX_NODES {
            self.accumulated_vote[node_id] =
                Self::apply_growth_to_value(params, self.accumulated_vote[node_id])?;
        }

        // === Step 2: vote credits from payload transactions ===
        if let Some(payload) = block.transactions() {
            for tx in payload.iter() {
                // FR51 replay-safety: zero-input UTXO carry-forward
                // contributes no vote credit.
                if let Some(complex) = tx.as_complex()
                    && complex.input_count() == 0
                {
                    continue;
                }
                let target = tx.vote() as usize;
                if target >= MAX_NODES {
                    // Out-of-bounds target: upstream validation (FR6) should
                    // have rejected. Defensively skip rather than panic.
                    continue;
                }
                self.accumulated_vote[target] = self.accumulated_vote[target]
                    .checked_add(vote_scale)
                    .ok_or(VoteEngineError::AccumulatedVoteOverflow)?;
            }
        }

        // === Step 3: creator reset (comes last, wipes any self-credit) ===
        let creator_idx = creator as usize;
        if creator_idx < MAX_NODES {
            self.accumulated_vote[creator_idx] = 0;
        }

        // === Deviation-block penalty (FR47) ===
        // A deviation block names a first-voted node other than its creator
        // (FR45). Its snapshot is verified by `check_header` before the apply;
        // the penalty resets the node whatever credits this block gave it.
        let victim = block.first_voted_node();
        if victim != creator && (victim as usize) < MAX_NODES {
            self.accumulated_vote[victim as usize] = 0;
        }

        Ok(())
    }

    /// Applies the exact inverse of [`apply_block`](Self::apply_block).
    ///
    /// Reverses the FR37 forward steps in reverse order (deviation penalty → creator
    /// reset → vote credits → anti-capture interest) using the two pre-reset
    /// snapshots stored in the block header:
    ///
    /// - `block.consumed_votes()` — the creator's post-interest, pre-credit value
    ///   (FR37, DEC-156).
    /// - `block.consumed_votes_from_first_voted_node()` — for a deviation block
    ///   (`first_voted_node != creator`), the penalized node's post-interest,
    ///   pre-credit value.
    ///
    /// Both snapshots precede the block's credits, and the reset or penalty wiped
    /// those credits, so neither node gets a credit subtraction: each is restored
    /// as the interest pre-image of its snapshot. The headers must be true
    /// ([`Self::check_header`]); the blockchain verifies them before applying.
    ///
    /// Interest rollback uses an exact monotonic inverse search for the configured
    /// capped growth function. If a post-interest value is not reachable, the method
    /// returns [`VoteEngineError::UnreachableInterestState`] rather than guessing.
    ///
    /// Atomic on failure: every node's post-undo value is preflighted before
    /// any slot is written (the same fail-closed discipline as
    /// [`apply_block`](Self::apply_block)), so an error leaves the
    /// accumulated-vote state unchanged.
    pub fn undo_block(&mut self, block: BlockView<'_>) -> Result<(), VoteEngineError> {
        let params = self.params()?;
        let creator_idx = block.creator() as usize;
        // Only a deviation block (`first_voted_node != creator`, FR45) penalized
        // a second node and recorded its snapshot.
        let first_voted = block.first_voted_node() as usize;
        let victim = (first_voted != creator_idx && first_voted < MAX_NODES)
            .then_some((first_voted, block.consumed_votes_from_first_voted_node()));

        // Preflight: compute every node's post-undo value without mutating, so
        // arithmetic errors fail closed and leave the state unchanged.
        for node_id in 0..MAX_NODES {
            self.undo_value_for(params, node_id, &block, creator_idx, victim)?;
        }

        // Commit: repeat the identical per-node computation, now writing. Each
        // node's value depends only on its own pre-undo slot plus the block,
        // so in-place writes cannot influence later nodes.
        for node_id in 0..MAX_NODES {
            self.accumulated_vote[node_id] =
                self.undo_value_for(params, node_id, &block, creator_idx, victim)?;
        }
        Ok(())
    }

    /// Computes one node's post-undo accumulated vote from the current state
    /// and the block header/payload, without mutating anything. Reverses the
    /// FR37 steps for that node in reverse order:
    ///
    /// - Creator and deviation victim: the slot restores from its pre-credit
    ///   snapshot (`consumed_votes` / `consumed_votes_from_first_voted_node`),
    ///   with no credit subtraction — the reset or penalty wiped the credits.
    /// - Every other node — step 2 rollback: subtract one `vote_scale` per
    ///   credit the block's payload gave it (FR51 zero-input carry-forward skip
    ///   mirrored).
    /// - Step 1 rollback: exact pre-image of the capped interest growth.
    fn undo_value_for(
        &self,
        params: VoteParams,
        node_id: usize,
        block: &BlockView<'_>,
        creator_idx: usize,
        victim: Option<(usize, u32)>,
    ) -> Result<u32, VoteEngineError> {
        let snapshot = match victim {
            _ if node_id == creator_idx => Some(block.consumed_votes()),
            Some((victim_idx, consumed)) if victim_idx == node_id => Some(consumed),
            _ => None,
        };
        if let Some(pre_credit) = snapshot {
            return Self::undo_growth_value(params, pre_credit);
        }

        let mut value = self.accumulated_vote[node_id];
        if let Some(payload) = block.transactions() {
            for tx in payload.iter() {
                if let Some(complex) = tx.as_complex()
                    && complex.input_count() == 0
                {
                    continue;
                }
                if tx.vote() as usize == node_id {
                    value = value
                        .checked_sub(params.vote_scale.get() as u32)
                        .ok_or(VoteEngineError::AccumulatedVoteUnderflow)?;
                }
            }
        }

        Self::undo_growth_value(params, value)
    }

    /// Checks a block's vote header fields against the current (parent) state:
    /// FR9 Tier 3 invariant I-09 (FR37, FR45; Story 6.2 D5, DEC-156). Read-only,
    /// and it reads no payload.
    ///
    /// - The creator's `consumed_votes` equals its post-interest, pre-credit value.
    /// - A normal block (`first_voted_node == creator`) carries
    ///   `consumed_votes_from_first_voted_node == 0`.
    /// - A deviation block names the creator-order top (FR38) and carries that
    ///   node's post-interest, pre-credit value.
    ///
    /// Each value is checked only where the state is authoritative. In a
    /// `complete` projection everything is checked. Otherwise, a node's own
    /// value is checked once `is_seeded` holds for it, and the top identity is
    /// not checked, because a partial seeding cannot tell who leads. The normal
    /// form (`consumed_votes_from_first_voted_node == 0` when the creator is the
    /// first-voted node) reads no state, so it is checked in every case; it is
    /// also the blockchain's FR9 Tier 1 header form.
    ///
    /// Returns `Ok(false)` on a mismatch (exact evidence for the caller), and
    /// `Err` only for [`VoteEngineError::NotParameterized`] or an interest
    /// overflow.
    pub fn check_header(
        &self,
        block: &BlockView<'_>,
        complete: bool,
        is_seeded: impl Fn(u32) -> bool,
    ) -> Result<bool, VoteEngineError> {
        let params = self.params()?;
        let snapshot =
            |node: u32| Self::apply_growth_to_value(params, self.accumulated_vote_of(node));
        let creator = block.creator();
        if (complete || is_seeded(creator)) && block.consumed_votes() != snapshot(creator)? {
            return Ok(false);
        }
        let first_voted = block.first_voted_node();
        let recorded = block.consumed_votes_from_first_voted_node();
        if first_voted == creator {
            return Ok(recorded == 0);
        }
        if complete && self.top_creator() != Some(first_voted) {
            return Ok(false);
        }
        if (complete || is_seeded(first_voted)) && recorded != snapshot(first_voted)? {
            return Ok(false);
        }
        Ok(true)
    }

    /// Seeds accumulated-vote state from a balance-block snapshot (FR50).
    ///
    /// Each `NodeInfoView` entry sets `accumulated_vote[owner] = vote_count`.
    /// Un-listed nodes are left untouched — Story 6.3 chain-switch replay
    /// re-forwards interest on top of the seeded baseline.
    ///
    /// If `block` is not a balance block (payload type != 2), this is a
    /// defensive no-op. Owner ids outside `MAX_NODES` are silently skipped.
    pub fn seed_from_balance_block(&mut self, block: BlockView<'_>) {
        let Some(payload) = block.balances() else {
            return;
        };
        for entry in payload.iter() {
            let owner = entry.owner() as usize;
            if owner >= MAX_NODES {
                continue;
            }
            self.accumulated_vote[owner] = entry.vote_count();
        }
    }

    /// Returns the accumulated vote for `node_id`. Returns `0` for
    /// out-of-bounds `node_id` (defensive).
    pub fn accumulated_vote_of(&self, node_id: u32) -> u32 {
        let idx = node_id as usize;
        if idx < MAX_NODES {
            self.accumulated_vote[idx]
        } else {
            0
        }
    }

    /// Returns the currently expected block creator per FR38.
    ///
    /// Ordering: descending by `accumulated_vote`, tie-broken by ascending
    /// `node_id`. Zero-vote nodes are ranked too (after every non-zero node),
    /// so the all-zero bootstrap state yields node 0 and the network can
    /// start. Returns `None` only in the degenerate `MAX_NODES == 0`
    /// configuration.
    ///
    /// Pure `&self` scan, O(MAX_NODES), uncached: iterate node ids in
    /// ascending order and keep the best under the (vote_desc, node_id_asc)
    /// order — replace only on a **strictly greater** vote so ties naturally
    /// break to the lower id.
    ///
    /// Deadline and grace-period progression are deliberately **not** this
    /// crate's concern: the blockchain module (Epic 8 / FR44–FR47) owns the
    /// deadline registry and, as the grace period expands, walks the same
    /// projection via [`creator_at_rank`](Self::creator_at_rank).
    pub fn top_creator(&self) -> Option<u32> {
        let mut top: Option<(u32, u32)> = None;
        for node_id in 0..MAX_NODES {
            let av = self.accumulated_vote[node_id];
            match top {
                None => top = Some((node_id as u32, av)),
                Some((_, top_vote)) if av > top_vote => top = Some((node_id as u32, av)),
                _ => {}
            }
        }
        top.map(|(node_id, _)| node_id)
    }

    /// Returns the `rank`-th creator in the FR38 descending-by-vote order.
    ///
    /// `rank == 0` is equivalent to `top_creator`. `rank == 1` is the
    /// second entry, and so on. The order is total over all `MAX_NODES`
    /// slots — zero-vote nodes form its tail in ascending-id order — so
    /// `None` is returned only when `rank >= MAX_NODES`.
    ///
    /// Not cached — O(N × (rank + 1)) via repeated linear scans. Epic 8 /
    /// FR44 walks the top few ranks for grace-period expansion; profiling
    /// will determine whether a sorted-index cache is worth adding later.
    pub fn creator_at_rank(&self, rank: usize) -> Option<u32> {
        // Track the previous rank's key so we can find the next one strictly
        // after it in the sort order. `None` before the first iteration means
        // "search from the top of the ordering".
        let mut cursor: Option<(u32, u32)> = None;
        for _ in 0..=rank {
            let mut best: Option<(u32, u32)> = None;
            for node_id in 0..MAX_NODES {
                let key = (self.accumulated_vote[node_id], node_id as u32);
                // Skip candidates that come at or before the cursor in the
                // (vote_desc, node_id_asc) ordering — those are already-ranked
                // entries.
                if let Some(cur) = cursor {
                    let strictly_after = key.0 < cur.0 || (key.0 == cur.0 && key.1 > cur.1);
                    if !strictly_after {
                        continue;
                    }
                }
                // Track the greatest candidate under the same ordering.
                match best {
                    None => best = Some(key),
                    Some(b) if key.0 > b.0 => best = Some(key),
                    Some(b) if key.0 == b.0 && key.1 < b.1 => best = Some(key),
                    _ => {}
                }
            }
            best?;
            cursor = best;
        }
        cursor.map(|(_, node_id)| node_id)
    }

    /// Returns whether `node_id` falls within the top-`rank` band of the
    /// FR38 creator order — `rank == 1` means "is the top creator",
    /// `rank == 2` means "top creator or second-eligible", and so on
    /// (`is_creator_within_rank(k, x)` ⇔ `creator_at_rank(r) == Some(x)` for
    /// some `r < k`).
    ///
    /// `rank == 0` denotes an empty band and always returns `false`, as do
    /// out-of-bounds `node_id` values. Zero-vote nodes are ranked (they form
    /// the ascending-id tail of the order), so at bootstrap the lowest node
    /// ids fill the band.
    ///
    /// Single O(MAX_NODES) scan with early exit: `node_id` is in the band iff
    /// fewer than `rank` nodes precede it in the (vote descending, node-id
    /// ascending) order. Intended as the frequent Epic 8 / FR44 admitted-set
    /// membership check.
    pub fn is_creator_within_rank(&self, rank: usize, node_id: u32) -> bool {
        let idx = node_id as usize;
        if rank == 0 || idx >= MAX_NODES {
            return false;
        }
        let own_vote = self.accumulated_vote[idx];
        // Count nodes strictly ahead of `node_id` in the
        // (vote_desc, node_id_asc) order; bail as soon as the band is full.
        let mut ahead = 0usize;
        for other_id in 0..MAX_NODES {
            if other_id == idx {
                continue;
            }
            let other_vote = self.accumulated_vote[other_id];
            if other_vote > own_vote || (other_vote == own_vote && other_id < idx) {
                ahead += 1;
                if ahead >= rank {
                    return false;
                }
            }
        }
        true
    }

    /// Test-only helper: directly set a node's accumulated vote. Not part
    /// of the runtime API; `seed_from_balance_block` is the runtime seeding path.
    #[cfg(test)]
    pub(crate) fn set_accumulated_vote_for_test(&mut self, node_id: u32, value: u32) {
        let idx = node_id as usize;
        if idx < MAX_NODES {
            self.accumulated_vote[idx] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonblokz_chain_types::{
        BlockBuilder, BlockHeader, ComplexTransaction, NodeInfo, NodeTransfer,
        PAYLOAD_TYPE_TRANSACTION,
    };
    use moonblokz_crypto::{Crypto, CryptoTrait, PRIVATE_KEY_SIZE};

    // Test instance parameters.
    const TEST_MAX_NODES: usize = 16;
    const TEST_VOTE_SCALE: u32 = 1000;
    const TEST_VOTE_INTEREST: u8 = 50; // 5% per block, capped at vote_scale

    type TestEngine = VoteEngine<TEST_MAX_NODES>;

    fn test_vote_scale() -> NonZeroU16 {
        NonZeroU16::new(TEST_VOTE_SCALE as u16).expect("test vote scale must be non-zero")
    }

    /// Constructs a real Schnorr (crypto-bigint) `Crypto` instance for
    /// signing test-fixture blocks.
    fn test_crypto() -> Crypto {
        let private_key = [1u8; PRIVATE_KEY_SIZE];
        Crypto::new(private_key)
            .ok()
            .expect("test private key must be accepted by the crypto-bigint Schnorr backend")
    }

    /// Builds a `BlockView<'a>` from the given creator + header fields +
    /// optional single transaction. `buffer` holds the serialized bytes for
    /// the lifetime of the view.
    fn make_block_view<'a>(
        creator: u32,
        consumed_votes: u32,
        first_voted_node: u32,
        consumed_votes_from_first_voted_node: u32,
        tx_bytes: Option<&[u8]>,
        buffer: &'a mut [u8; moonblokz_chain_types::MAX_BLOCK_SIZE],
        crypto: &Crypto,
    ) -> BlockView<'a> {
        let header = BlockHeader {
            version: 1,
            sequence: 0,
            creator,
            mined_amount: 0,
            payload_type: PAYLOAD_TYPE_TRANSACTION,
            consumed_votes,
            first_voted_node,
            consumed_votes_from_first_voted_node,
            previous_hash: [0u8; 32],
            signature: [0u8; 64],
        };
        let mut builder = BlockBuilder::new().header(header);
        if let Some(bytes) = tx_bytes {
            builder
                .add_transaction_bytes(bytes)
                .ok()
                .expect("test tx must fit in the block payload");
        }
        let block = builder
            .build_signed(crypto)
            .ok()
            .expect("test block must build with the fixture crypto handle");
        let serialized = block.serialized_bytes();
        buffer[..serialized.len()].copy_from_slice(serialized);
        BlockView::from_bytes(&buffer[..serialized.len()])
            .ok()
            .expect("just-built block must parse as BlockView")
    }

    fn make_empty_block<'a>(
        creator: u32,
        buffer: &'a mut [u8; moonblokz_chain_types::MAX_BLOCK_SIZE],
        crypto: &Crypto,
    ) -> BlockView<'a> {
        make_block_view(creator, 0, creator, 0, None, buffer, crypto)
    }

    /// Builds a balance-block `BlockView<'a>` containing the given node-info
    /// entries. `buffer` holds the serialized bytes for the lifetime of the
    /// view.
    fn make_balance_block<'a>(
        entries: &[NodeInfo],
        buffer: &'a mut [u8; moonblokz_chain_types::MAX_BLOCK_SIZE],
        crypto: &Crypto,
    ) -> BlockView<'a> {
        let header = BlockHeader {
            version: 1,
            sequence: 0,
            creator: 0,
            mined_amount: 0,
            // Left at 0 so `add_node_info` can set it to PAYLOAD_TYPE_BALANCE.
            payload_type: 0,
            consumed_votes: 0,
            first_voted_node: 0,
            consumed_votes_from_first_voted_node: 0,
            previous_hash: [0u8; 32],
            signature: [0u8; 64],
        };
        let mut builder = BlockBuilder::new().header(header);
        for ni in entries {
            builder
                .add_node_info(ni)
                .ok()
                .expect("test node-info must fit in the balance block payload");
        }
        // `set_max_node_id` is optional but keeps the balance header well-formed.
        builder
            .set_max_node_id(TEST_MAX_NODES as u32)
            .ok()
            .expect("set_max_node_id must succeed on a valid balance block");
        let block = builder
            .build_signed(crypto)
            .ok()
            .expect("test balance block must build with the fixture crypto handle");
        let serialized = block.serialized_bytes();
        buffer[..serialized.len()].copy_from_slice(serialized);
        BlockView::from_bytes(&buffer[..serialized.len()])
            .ok()
            .expect("just-built balance block must parse as BlockView")
    }

    /// Node-transfer tx with `vote_target = vote`, `initializer = 1`.
    fn nt_with_vote(vote: u32) -> NodeTransfer {
        let sig = [0xAA; 64];
        NodeTransfer::new(vote, 0, 1, 0, 100, 1, 0, &sig)
    }

    /// `init`'s field writes (`accumulated_vote` filled via `write_bytes`, the
    /// parameters written as a real `None`) must land every field in its
    /// correct default state — verified directly rather than trusted by
    /// construction. The engine starts **unparameterized**.
    ///
    /// The destructuring below is deliberately exhaustive (no `..`): adding a
    /// field to `VoteEngine` makes this test fail to *compile* until the new
    /// field is both initialized and asserted here, instead of silently
    /// leaving it unchecked. Never compare two engines byte-for-byte for this
    /// purpose: padding bytes are uninitialized, and Miri rejects reading them.
    #[test]
    fn init_sets_expected_defaults() {
        let mut slot = core::mem::MaybeUninit::<TestEngine>::uninit();
        let engine = TestEngine::init(&mut slot);

        let VoteEngine {
            accumulated_vote,
            params,
        } = &*engine;

        assert!(accumulated_vote.iter().all(|&v| v == 0));
        assert!(params.is_none(), "a fresh engine holds no FR37 parameters");
        // All-zero order is headed by node 0 (bootstrap rule) — the queries
        // need no parameters.
        assert_eq!(engine.top_creator(), Some(0));
    }

    /// `init` must produce exactly what the by-value specification `new()`
    /// produces. Both sides are destructured exhaustively (no `..`), so a new
    /// field cannot slip past either the literal in `new()` or this
    /// comparison. Never compare the two byte-for-byte: padding is
    /// uninitialized, and Miri rejects reading it.
    #[test]
    fn init_is_equivalent_to_new() {
        let mut slot = core::mem::MaybeUninit::<TestEngine>::uninit();
        let built = TestEngine::init(&mut slot);
        let spec = TestEngine::new();

        let VoteEngine {
            accumulated_vote,
            params,
        } = &*built;
        let VoteEngine {
            accumulated_vote: spec_accumulated_vote,
            params: spec_params,
        } = &spec;

        assert!(accumulated_vote == spec_accumulated_vote);
        assert!(params == spec_params);
        assert!(*built == spec);
    }

    /// `reset` must leave a used engine indistinguishable from a freshly
    /// `init`ed one with the same parameters.
    #[test]
    fn reset_restores_fresh_defaults() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.accumulated_vote[3] = 42;

        engine.reset(test_vote_scale(), TEST_VOTE_INTEREST);

        let fresh = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let VoteEngine {
            accumulated_vote,
            params,
        } = &engine;
        assert!(accumulated_vote.iter().all(|&v| v == 0));
        assert!(params == &fresh.params);
    }

    /// Without FR37 parameters every vote effect refuses, and refuses without
    /// touching the accumulated-vote state. The queries stay answerable.
    #[test]
    fn unparameterized_engine_refuses_vote_effects() {
        let mut slot = core::mem::MaybeUninit::<TestEngine>::uninit();
        let engine = TestEngine::init(&mut slot);
        let crypto = test_crypto();
        let mut buf_a = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let mut buf_b = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let mut buf_c = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];

        assert_eq!(
            engine.apply_block(make_empty_block(3, &mut buf_a, &crypto)),
            Err(VoteEngineError::NotParameterized)
        );
        assert_eq!(
            engine.undo_block(make_empty_block(3, &mut buf_b, &crypto)),
            Err(VoteEngineError::NotParameterized)
        );
        assert!(engine.accumulated_vote.iter().all(|&v| v == 0));
        assert_eq!(engine.top_creator(), Some(0));

        // Parameterizing is the one transition that makes the engine usable.
        engine.reset(test_vote_scale(), TEST_VOTE_INTEREST);
        assert!(
            engine
                .apply_block(make_empty_block(3, &mut buf_c, &crypto))
                .is_ok()
        );
    }

    /// `clear` empties the working set without needing parameters — the FR3 /
    /// FR5 "clean working set" step for a node that holds no configuration.
    #[test]
    fn clear_empties_without_parameters() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.accumulated_vote[2] = 7;

        engine.clear();

        assert!(engine.accumulated_vote.iter().all(|&v| v == 0));
        assert!(engine.params.is_some(), "clear leaves the parameters alone");
    }

    #[test]
    fn apply_block_step1_zero_stays_zero() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(3, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        for i in 0..TEST_MAX_NODES {
            assert_eq!(engine.accumulated_vote[i], 0);
        }
    }

    #[test]
    fn apply_block_step1_interest_then_creator_reset() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 2000); // non-creator
        engine.set_accumulated_vote_for_test(5, 5000); // non-creator
        engine.set_accumulated_vote_for_test(3, 3000); // creator

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(3, &mut buf, &crypto); // creator = 3

        engine.apply_block(block).expect("apply block must succeed");

        // 2000 + min(floor(2000 * 50 / 1000), 1000) = 2100
        assert_eq!(engine.accumulated_vote[2], 2100);
        // 5000 + min(floor(5000 * 50 / 1000), 1000) = 5250
        assert_eq!(engine.accumulated_vote[5], 5250);
        // Creator also receives step-1 interest, then step 3 resets it to 0.
        assert_eq!(engine.accumulated_vote[3], 0);
    }

    #[test]
    fn apply_block_interest_bump_is_capped_at_vote_scale() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        // raw bump = floor(40_000 * 50 / 1000) = 2000, capped to vote_scale = 1000.
        engine.set_accumulated_vote_for_test(2, 40_000);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(3, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        assert_eq!(engine.accumulated_vote[2], 41_000);
    }

    #[test]
    fn apply_block_reports_overflow_instead_of_saturating() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, u32::MAX - 999);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(3, &mut buf, &crypto);

        let result = engine.apply_block(block);

        assert_eq!(result, Err(VoteEngineError::AccumulatedVoteOverflow));
        assert_eq!(engine.accumulated_vote[2], u32::MAX - 999);
    }

    #[test]
    fn apply_block_step2_vote_credit() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let nt = nt_with_vote(5);
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 0, 3, 0, Some(nt.as_bytes()), &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        assert_eq!(engine.accumulated_vote[5], TEST_VOTE_SCALE);
    }

    #[test]
    fn apply_block_step2_node0_permanent_target() {
        // Creator = 3 (not 0), so step 3 does NOT wipe node 0.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let nt = nt_with_vote(0);
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 0, 3, 0, Some(nt.as_bytes()), &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        assert_eq!(engine.accumulated_vote[0], TEST_VOTE_SCALE);
    }

    #[test]
    fn apply_block_step2_zero_input_utxo_no_credit() {
        // Complex tx with `input_count == 0` (carry-forward) contributes no
        // vote credit even if `vote != 0`.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let complex = ComplexTransaction::new(7);
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 0, 3, 0, Some(complex.as_bytes()), &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        assert_eq!(engine.accumulated_vote[7], 0);
    }

    #[test]
    fn apply_block_step3_creator_reset() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(3, 8000);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(3, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        assert_eq!(engine.accumulated_vote[3], 0);
    }

    #[test]
    fn apply_block_creator_self_vote_does_not_retain() {
        // Creator = 3, tx votes for node 3 (self). Step 3 wipes step 2.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let nt = nt_with_vote(3);
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 0, 3, 0, Some(nt.as_bytes()), &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        assert_eq!(engine.accumulated_vote[3], 0);
    }

    #[test]
    fn apply_block_deviation_penalty() {
        // Deviation block: creator = 5, first_voted_node = 2,
        // consumed_votes_from_first_voted_node = 1. Node 2 reset.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 4000);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(5, 0, 2, 1, None, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        assert_eq!(engine.accumulated_vote[2], 0);
    }

    #[test]
    fn apply_block_ordinary_block_no_deviation_penalty() {
        // Ordinary block: `first_voted_node == creator` — no deviation reset.
        // Node 4 gets the interest bump instead.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(4, 4000);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(5, 0, 5, 0, None, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");

        // 4000 + min(floor(4000 * 50 / 1000), 1000) = 4200
        assert_eq!(engine.accumulated_vote[4], 4200);
    }

    #[test]
    fn apply_block_deterministic_replay() {
        // Two engines fed the identical block sequence must reach identical state.
        fn run() -> [u32; TEST_MAX_NODES] {
            let crypto = test_crypto();
            let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);

            let mut buf1 = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
            let nt1 = nt_with_vote(5);
            let block1 = make_block_view(3, 0, 3, 0, Some(nt1.as_bytes()), &mut buf1, &crypto);
            engine
                .apply_block(block1)
                .expect("apply block must succeed");

            let mut buf2 = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
            let nt2 = nt_with_vote(5);
            let block2 = make_block_view(7, 0, 7, 0, Some(nt2.as_bytes()), &mut buf2, &crypto);
            engine
                .apply_block(block2)
                .expect("apply block must succeed");

            engine.accumulated_vote
        }

        let a = run();
        let b = run();
        assert_eq!(a, b, "identical op sequence must yield identical state");
    }

    #[test]
    fn accumulated_vote_of_out_of_bounds_returns_zero() {
        let engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        assert_eq!(engine.accumulated_vote_of(TEST_MAX_NODES as u32), 0);
        assert_eq!(engine.accumulated_vote_of(u32::MAX), 0);
    }

    // ==================================================================
    // Story 3.2 — undo_block & seed_from_balance_block
    // ==================================================================

    #[test]
    fn undo_reverses_step3_creator_reset() {
        // Post-interest value = 2000 + min(floor(2000 * 50 / 1000), 1000) = 2100.
        // The block header records that pre-reset value in `consumed_votes`.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(3, 2000);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 2100, 3, 0, None, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[3], 0);

        // Re-parse the buffer so the view is not borrowed by the applied block.
        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");

        assert_eq!(engine.accumulated_vote[3], 2000);
    }

    #[test]
    fn undo_reverses_step2_vote_credit() {
        // Apply a block whose payload credits node 5 (via a NodeTransfer with
        // vote = 5). Undo must subtract the credit.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let nt = nt_with_vote(5);
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 0, 3, 0, Some(nt.as_bytes()), &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[5], TEST_VOTE_SCALE);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");

        assert_eq!(engine.accumulated_vote[5], 0);
    }

    #[test]
    fn undo_reverses_step1_interest_values() {
        // Apply an empty block, then undo — nodes must be exactly restored.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 2000);
        engine.set_accumulated_vote_for_test(5, 5000);
        engine.set_accumulated_vote_for_test(7, 100);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        // Creator = 4 (uninhabited so step 3 is a no-op at av[4] = 0).
        let block = make_empty_block(4, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");
        // Post-interest values:
        assert_eq!(engine.accumulated_vote[2], 2100);
        assert_eq!(engine.accumulated_vote[5], 5250);
        assert_eq!(engine.accumulated_vote[7], 105);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");

        assert_eq!(engine.accumulated_vote[2], 2000);
        assert_eq!(engine.accumulated_vote[5], 5000);
        assert_eq!(engine.accumulated_vote[7], 100);
    }

    #[test]
    fn undo_reverses_step1_interest_non_multiple_values() {
        // Values that used to fail under the old closed-form floor inverse now
        // round-trip through the exact monotonic inverse search.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 19); // bump 0, reachable after stays 19
        engine.set_accumulated_vote_for_test(5, 21); // bump 1, reachable after is 22

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(4, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[2], 19);
        assert_eq!(engine.accumulated_vote[5], 22);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");

        assert_eq!(engine.accumulated_vote[2], 19);
        assert_eq!(engine.accumulated_vote[5], 21);
    }

    #[test]
    fn undo_reverses_step1_interest_capped_region() {
        // cap_threshold = ceil(1000^2 / 50) = 20_000 with the test params.
        // Covers the capped linear region (fast-path inverse), the exact
        // fast-path boundary (`after == cap_threshold + vote_scale`), and a
        // just-below-threshold value that must take the binary-search path.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 20_000); // at threshold: bump capped to 1000
        engine.set_accumulated_vote_for_test(5, 25_000); // inside capped region
        engine.set_accumulated_vote_for_test(7, 19_999); // below threshold: raw bump 999

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(4, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[2], 21_000);
        assert_eq!(engine.accumulated_vote[5], 26_000);
        assert_eq!(engine.accumulated_vote[7], 20_998);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");

        assert_eq!(engine.accumulated_vote[2], 20_000);
        assert_eq!(engine.accumulated_vote[5], 25_000);
        assert_eq!(engine.accumulated_vote[7], 19_999);
    }

    #[test]
    fn undo_handles_vote_interest_above_vote_scale() {
        // Degenerate config: vote_interest (255) > vote_scale (2), so
        // cap_threshold = ceil(2^2 / 255) = 1 and bump(1) = vote_scale = 2
        // exceeds the value itself. The tightened search lower bound
        // `after - bump(after)` must saturate at zero instead of underflowing.
        let scale = NonZeroU16::new(2).expect("test vote scale must be non-zero");
        let mut engine = VoteEngine::<TEST_MAX_NODES>::new_for_test(scale, 255);
        let crypto = test_crypto();

        // Reachable round trip: 1 -> 3 (bump capped at vote_scale) -> 1.
        engine.set_accumulated_vote_for_test(5, 1);
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(4, &mut buf, &crypto);
        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[5], 3);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");
        assert_eq!(engine.accumulated_vote[5], 1);

        // `after = 1` is unreachable (f(0) = 0, f(1) = 3): the search must
        // report it — without underflow panic on `after - bump(after)`.
        let mut buf2 = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block2 = make_empty_block(4, &mut buf2, &crypto);
        let result = engine.undo_block(block2);
        assert_eq!(result, Err(VoteEngineError::UnreachableInterestState));
    }

    #[test]
    fn undo_rejects_unreachable_interest_state() {
        // With bump(x) = floor(x / 20), f(1959) = 2056 and f(1960) = 2058:
        // 2057 has no pre-image under the configured growth function, so the
        // interest rollback must report it instead of guessing a nearby value.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 2057);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        make_empty_block(4, &mut buf, &crypto);
        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");

        let result = engine.undo_block(block_for_undo);

        assert_eq!(result, Err(VoteEngineError::UnreachableInterestState));
    }

    #[test]
    fn undo_unreachable_error_leaves_state_unchanged() {
        // av[5] = 2057 has no interest pre-image; av[2] = 5000 does. The
        // preflight must reject the block before any slot (including the
        // reachable av[2]) is rewritten.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 5000);
        engine.set_accumulated_vote_for_test(5, 2057);
        let pre_state = engine.accumulated_vote;

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(4, &mut buf, &crypto);

        let result = engine.undo_block(block);

        assert_eq!(result, Err(VoteEngineError::UnreachableInterestState));
        assert_eq!(
            engine.accumulated_vote, pre_state,
            "failed undo must not mutate state"
        );
    }

    #[test]
    fn undo_underflow_error_leaves_state_unchanged() {
        // The block's payload credits node 5, but av[5] = 500 < vote_scale, so
        // the step-2 rollback underflows. Every slot must stay untouched.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 5000);
        engine.set_accumulated_vote_for_test(5, 500);
        let pre_state = engine.accumulated_vote;

        let crypto = test_crypto();
        let nt = nt_with_vote(5);
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 0, 3, 0, Some(nt.as_bytes()), &mut buf, &crypto);

        let result = engine.undo_block(block);

        assert_eq!(result, Err(VoteEngineError::AccumulatedVoteUnderflow));
        assert_eq!(
            engine.accumulated_vote, pre_state,
            "failed undo must not mutate state"
        );
    }

    #[test]
    fn undo_reverses_deviation_penalty() {
        // Deviation block: creator = 5, first_voted_node = 3.
        // Pre-apply av[3] = 4000. Forward step 1 bumps it to 4200, the
        // post-interest, pre-credit snapshot recorded in
        // `consumed_votes_from_first_voted_node` (DEC-156), before the penalty
        // wipes it. Undo takes the interest pre-image of the snapshot: 4000.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(3, 4000);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(5, 0, 3, 4200, None, &mut buf, &crypto);

        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[3], 0);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");

        assert_eq!(engine.accumulated_vote[3], 4000);
    }

    #[test]
    fn apply_undo_round_trip_multi_block() {
        // The exact monotonic inverse supports arbitrary reachable values across
        // multiple blocks, not only values that satisfy a divisibility invariant.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 8000);
        let pre_state = engine.accumulated_vote;

        let crypto = test_crypto();
        // Block A: creator 4, empty.
        let mut buf_a = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block_a = make_empty_block(4, &mut buf_a, &crypto);
        engine
            .apply_block(block_a)
            .expect("apply block must succeed");

        // Block B: creator 6, empty.
        let mut buf_b = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block_b = make_empty_block(6, &mut buf_b, &crypto);
        engine
            .apply_block(block_b)
            .expect("apply block must succeed");

        // Block C: creator 8, empty.
        let mut buf_c = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block_c = make_empty_block(8, &mut buf_c, &crypto);
        engine
            .apply_block(block_c)
            .expect("apply block must succeed");

        // Undo in reverse: C, B, A.
        let block_c_undo = BlockView::from_bytes(&buf_c[..])
            .ok()
            .expect("block C buffer must reparse");
        engine
            .undo_block(block_c_undo)
            .expect("undo block must succeed");

        let block_b_undo = BlockView::from_bytes(&buf_b[..])
            .ok()
            .expect("block B buffer must reparse");
        engine
            .undo_block(block_b_undo)
            .expect("undo block must succeed");

        let block_a_undo = BlockView::from_bytes(&buf_a[..])
            .ok()
            .expect("block A buffer must reparse");
        engine
            .undo_block(block_a_undo)
            .expect("undo block must succeed");

        assert_eq!(
            engine.accumulated_vote, pre_state,
            "apply→undo round-trip must restore the pre-state"
        );
    }

    #[test]
    fn seed_from_balance_block_populates_snapshots() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let pk = [0xBBu8; 32];
        let entries = [
            NodeInfo::new(5, 0, 2000, &pk),
            NodeInfo::new(7, 0, 5000, &pk),
            NodeInfo::new(11, 0, 100, &pk),
        ];
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_balance_block(&entries, &mut buf, &crypto);

        engine.seed_from_balance_block(block);

        assert_eq!(engine.accumulated_vote[5], 2000);
        assert_eq!(engine.accumulated_vote[7], 5000);
        assert_eq!(engine.accumulated_vote[11], 100);
        // Other slots untouched.
        for i in 0..TEST_MAX_NODES {
            if i == 5 || i == 7 || i == 11 {
                continue;
            }
            assert_eq!(engine.accumulated_vote[i], 0, "slot {i} must remain zero");
        }
    }

    #[test]
    fn seed_from_balance_block_ignores_non_balance_block() {
        // Pass a transaction block — `balances()` returns None so the seed is
        // a defensive no-op. Pre-existing state stays unchanged.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 4000);

        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(3, &mut buf, &crypto);
        // Sanity: this is a transaction block, not a balance block.
        assert!(block.balances().is_none());

        engine.seed_from_balance_block(block);

        assert_eq!(engine.accumulated_vote[2], 4000);
    }

    #[test]
    fn seed_from_balance_block_bounds_checks_owner() {
        // Owner id at TEST_MAX_NODES is out of bounds — must be silently
        // skipped without panicking, and in-bounds slots stay unchanged.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 4000);

        let crypto = test_crypto();
        let pk = [0xBBu8; 32];
        let entries = [
            NodeInfo::new(TEST_MAX_NODES as u32, 0, 9999, &pk),
            NodeInfo::new(u32::MAX, 0, 42, &pk),
        ];
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_balance_block(&entries, &mut buf, &crypto);

        engine.seed_from_balance_block(block);

        // Other slots untouched (defensive skip).
        assert_eq!(engine.accumulated_vote[2], 4000);
        for i in 0..TEST_MAX_NODES {
            if i == 2 {
                continue;
            }
            assert_eq!(engine.accumulated_vote[i], 0, "slot {i} must remain zero");
        }
    }

    // ==================================================================
    // Story 3.3 — top_creator & creator_at_rank (FR38)
    // ==================================================================

    #[test]
    fn top_creator_all_zero_bootstrap_starts_at_node_zero() {
        // Bootstrap: every accumulated vote is zero, yet the network must be
        // able to start — the order degrades to pure ascending node id, so
        // node 0 (genesis) is the expected creator and low ids fill the band.
        let engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        assert_eq!(engine.top_creator(), Some(0));
        assert_eq!(engine.creator_at_rank(0), Some(0));
        assert_eq!(engine.creator_at_rank(1), Some(1));
        assert!(engine.is_creator_within_rank(1, 0));
        assert!(!engine.is_creator_within_rank(1, 1));
        assert!(engine.is_creator_within_rank(2, 1));
    }

    #[test]
    fn top_creator_single_node_returns_that_node() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(5, 100);
        assert_eq!(engine.top_creator(), Some(5));
    }

    #[test]
    fn top_creator_returns_highest_accumulated_vote() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 100);
        engine.set_accumulated_vote_for_test(5, 300);
        engine.set_accumulated_vote_for_test(7, 200);
        assert_eq!(engine.top_creator(), Some(5));
    }

    #[test]
    fn top_creator_ties_broken_by_ascending_node_id() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        // Equal votes on nodes 2, 5, 11 — node 2 (lowest id) must win.
        engine.set_accumulated_vote_for_test(2, 500);
        engine.set_accumulated_vote_for_test(5, 500);
        engine.set_accumulated_vote_for_test(11, 500);
        assert_eq!(engine.top_creator(), Some(2));
    }

    #[test]
    fn top_creator_ranks_nonzero_vote_above_zero_vote() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        // Only node 9 is non-zero — it outranks every zero-vote node,
        // including the lower ids.
        engine.set_accumulated_vote_for_test(9, 42);
        assert_eq!(engine.top_creator(), Some(9));
    }

    #[test]
    fn top_creator_tracks_state_after_apply_block() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(5, 200);
        engine.set_accumulated_vote_for_test(7, 100);
        assert_eq!(engine.top_creator(), Some(5));

        // Apply a block where node 5 is the creator — step 3 resets av[5] to 0.
        // After apply: av[5] = 0, av[7] = 105 (interest bump). Top becomes 7.
        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(5, &mut buf, &crypto);
        engine.apply_block(block).expect("apply block must succeed");

        assert_eq!(engine.top_creator(), Some(7));
    }

    #[test]
    fn top_creator_tracks_state_after_undo_block() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(3, 2000);
        engine.set_accumulated_vote_for_test(7, 100);
        assert_eq!(engine.top_creator(), Some(3));

        // Apply a block where node 3 is creator (resets av[3] to 0). Then
        // undo it — av[3] should be restored to 2000, and top_creator should
        // recompute back to 3.
        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 2100, 3, 0, None, &mut buf, &crypto);
        engine.apply_block(block).expect("apply block must succeed");
        // After apply: av[3] = 0, av[7] = 105 → top = 7.
        assert_eq!(engine.top_creator(), Some(7));

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");

        // After undo: av[3] = 2000, av[7] = 100 → top = 3 again.
        assert_eq!(engine.top_creator(), Some(3));
    }

    #[test]
    fn top_creator_tracks_state_after_seed_from_balance_block() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 100);
        assert_eq!(engine.top_creator(), Some(2));

        // Seed a balance block that reseeds node 5 with a much larger vote —
        // top should reflect node 5 afterwards.
        let crypto = test_crypto();
        let pk = [0xBBu8; 32];
        let entries = [NodeInfo::new(5, 0, 9999, &pk)];
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_balance_block(&entries, &mut buf, &crypto);
        engine.seed_from_balance_block(block);

        assert_eq!(engine.top_creator(), Some(5));
    }

    #[test]
    fn creator_at_rank_matches_descending_order() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 100);
        engine.set_accumulated_vote_for_test(5, 300);
        engine.set_accumulated_vote_for_test(7, 200);

        assert_eq!(engine.creator_at_rank(0), Some(5)); // vote 300
        assert_eq!(engine.creator_at_rank(1), Some(7)); // vote 200
        assert_eq!(engine.creator_at_rank(2), Some(2)); // vote 100
    }

    #[test]
    fn creator_at_rank_covers_zero_vote_tail_and_ends_at_max_nodes() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 100);
        engine.set_accumulated_vote_for_test(5, 300);
        engine.set_accumulated_vote_for_test(7, 200);
        // Ranks 0-2 are the non-zero nodes; the zero-vote tail follows in
        // ascending-id order (0, 1, 3, 4, ...).
        assert_eq!(engine.creator_at_rank(3), Some(0));
        assert_eq!(engine.creator_at_rank(4), Some(1));
        assert_eq!(engine.creator_at_rank(5), Some(3));
        // The order is total over all MAX_NODES slots and ends there.
        assert_eq!(engine.creator_at_rank(TEST_MAX_NODES - 1), Some(15));
        assert_eq!(engine.creator_at_rank(TEST_MAX_NODES), None);
    }

    #[test]
    fn creator_at_rank_zero_matches_top_creator() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 100);
        engine.set_accumulated_vote_for_test(5, 500);
        engine.set_accumulated_vote_for_test(9, 500); // tied with 5, higher id

        assert_eq!(engine.creator_at_rank(0), engine.top_creator());
        // And that shared answer is the lowest-id tie-winner among the top-vote group.
        assert_eq!(engine.creator_at_rank(0), Some(5));
    }

    #[test]
    fn creator_at_rank_ties_broken_by_ascending_node_id() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 500);
        engine.set_accumulated_vote_for_test(5, 500);
        engine.set_accumulated_vote_for_test(11, 500);

        // Descending vote (all equal), tie-break ascending node id.
        assert_eq!(engine.creator_at_rank(0), Some(2));
        assert_eq!(engine.creator_at_rank(1), Some(5));
        assert_eq!(engine.creator_at_rank(2), Some(11));
        // The zero-vote tail starts right after: lowest zero-vote id is 0.
        assert_eq!(engine.creator_at_rank(3), Some(0));
    }

    #[test]
    fn is_creator_within_rank_top_band() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 100);
        engine.set_accumulated_vote_for_test(5, 300);
        engine.set_accumulated_vote_for_test(7, 200);

        // rank 1 == "is the top creator".
        assert!(engine.is_creator_within_rank(1, 5));
        assert!(!engine.is_creator_within_rank(1, 7));
        assert!(!engine.is_creator_within_rank(1, 2));
    }

    #[test]
    fn is_creator_within_rank_band_membership() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 100);
        engine.set_accumulated_vote_for_test(5, 300);
        engine.set_accumulated_vote_for_test(7, 200);

        // rank 2 band = {5, 7}.
        assert!(engine.is_creator_within_rank(2, 5));
        assert!(engine.is_creator_within_rank(2, 7));
        assert!(!engine.is_creator_within_rank(2, 2));
        // rank 3 band covers all three eligible nodes.
        assert!(engine.is_creator_within_rank(3, 2));
    }

    #[test]
    fn is_creator_within_rank_ties_follow_ascending_node_id() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(2, 500);
        engine.set_accumulated_vote_for_test(5, 500);
        engine.set_accumulated_vote_for_test(11, 500);

        // Order under the tie-break: 2, 5, 11.
        assert!(engine.is_creator_within_rank(1, 2));
        assert!(!engine.is_creator_within_rank(1, 5));
        assert!(engine.is_creator_within_rank(2, 5));
        assert!(!engine.is_creator_within_rank(2, 11));
        assert!(engine.is_creator_within_rank(3, 11));
    }

    #[test]
    fn is_creator_within_rank_zero_vote_tail_and_out_of_bounds() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(5, 100);

        // Zero-vote nodes rank after the non-zero node in ascending-id
        // order: 5, then 0, 1, 2, 3, 4, 6, 7, 8, 9, ... — node 9 sits at
        // 0-based position 9, so it needs a band of at least 10.
        assert!(engine.is_creator_within_rank(TEST_MAX_NODES, 9));
        assert!(!engine.is_creator_within_rank(9, 9));
        assert!(engine.is_creator_within_rank(10, 9));
        // Out-of-bounds ids are never in any band.
        assert!(!engine.is_creator_within_rank(1, TEST_MAX_NODES as u32));
        assert!(!engine.is_creator_within_rank(1, u32::MAX));
    }

    #[test]
    fn is_creator_within_rank_zero_rank_is_empty_band() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(5, 100);
        // rank 0 = empty band — even the top creator is outside it.
        assert!(!engine.is_creator_within_rank(0, 5));
    }

    #[test]
    fn is_creator_within_rank_one_matches_top_creator_after_writes() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(5, 300);
        engine.set_accumulated_vote_for_test(7, 200);

        // rank-1 band membership == "is the top creator".
        assert!(engine.is_creator_within_rank(1, 5));
        assert!(!engine.is_creator_within_rank(1, 7));

        // A write that changes the top flips membership accordingly.
        engine.set_accumulated_vote_for_test(7, 400);
        assert!(engine.is_creator_within_rank(1, 7));
        assert!(!engine.is_creator_within_rank(1, 5));
    }

    #[test]
    fn is_creator_within_rank_matches_creator_at_rank() {
        // Cross-check the O(N) membership scan against the rank walk:
        // membership in the top-k band ⇔ creator_at_rank(r) hits the node
        // for some r < k.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(1, 50);
        engine.set_accumulated_vote_for_test(3, 500);
        engine.set_accumulated_vote_for_test(6, 500);
        engine.set_accumulated_vote_for_test(9, 200);

        for band in 0..=5usize {
            for node in 0..TEST_MAX_NODES {
                let expected = (0..band).any(|r| engine.creator_at_rank(r) == Some(node as u32));
                assert_eq!(
                    engine.is_creator_within_rank(band, node as u32),
                    expected,
                    "band={band} node={node}"
                );
            }
        }
    }
    // ==================================================================
    // Story 6.2 — pre-credit header snapshots, `check_header` (I-09)
    // ==================================================================

    /// Deviation block: creator 5 credited by its own transaction, first-voted
    /// node 3 credited too; both snapshots are pre-credit.
    fn deviation_with_credits<'a>(
        buf: &'a mut [u8; moonblokz_chain_types::MAX_BLOCK_SIZE],
        crypto: &Crypto,
        consumed: u32,
        recorded: u32,
    ) -> BlockView<'a> {
        let nt_self = nt_with_vote(5);
        let nt_victim = nt_with_vote(3);
        let header = BlockHeader {
            version: 1,
            sequence: 0,
            creator: 5,
            mined_amount: 0,
            payload_type: PAYLOAD_TYPE_TRANSACTION,
            consumed_votes: consumed,
            first_voted_node: 3,
            consumed_votes_from_first_voted_node: recorded,
            previous_hash: [0u8; 32],
            signature: [0u8; 64],
        };
        let mut builder = BlockBuilder::new().header(header);
        for nt in [&nt_self, &nt_victim] {
            builder
                .add_transaction_bytes(nt.as_bytes())
                .ok()
                .expect("test tx must fit in the block payload");
        }
        let block = builder
            .build_signed(crypto)
            .ok()
            .expect("test block must build with the fixture crypto handle");
        let serialized = block.serialized_bytes();
        buf[..serialized.len()].copy_from_slice(serialized);
        BlockView::from_bytes(&buf[..serialized.len()])
            .ok()
            .expect("just-built block must parse as BlockView")
    }

    #[test]
    fn deviation_credits_are_wiped_and_undo_restores_from_snapshots() {
        // av[3] = 4000 (top), av[5] = 2000. Snapshots: I(2000) = 2100,
        // I(4000) = 4200. Both nodes also receive a credit in the block, which
        // the reset and the penalty wipe; undo restores both exactly.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(3, 4000);
        engine.set_accumulated_vote_for_test(5, 2000);
        engine.set_accumulated_vote_for_test(7, 300);
        let pre_state = engine.accumulated_vote;
        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = deviation_with_credits(&mut buf, &crypto, 2100, 4200);
        assert_eq!(engine.check_header(&block, true, |_| true), Ok(true));

        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[5], 0);
        assert_eq!(engine.accumulated_vote[3], 0);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");
        assert_eq!(engine.accumulated_vote, pre_state);
    }

    #[test]
    fn penalty_applies_when_the_first_voted_node_holds_no_vote() {
        // The first-voted node starts at 0, so its snapshot is 0, but the
        // block credits it: the penalty must still wipe that credit, and undo
        // must restore 0.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = deviation_with_credits(&mut buf, &crypto, 0, 0);
        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[3], 0);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");
        assert_eq!(engine.accumulated_vote[3], 0);
        assert_eq!(engine.accumulated_vote[5], 0);
    }

    #[test]
    fn creator_self_credit_round_trips_through_undo() {
        // A normal block whose own transaction credits its creator: the reset
        // wipes the credit; undo must not subtract it from the snapshot.
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(3, 2000);
        let crypto = test_crypto();
        let nt = nt_with_vote(3);
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 2100, 3, 0, Some(nt.as_bytes()), &mut buf, &crypto);
        assert_eq!(engine.check_header(&block, true, |_| true), Ok(true));
        engine.apply_block(block).expect("apply block must succeed");
        assert_eq!(engine.accumulated_vote[3], 0);

        let block_for_undo = BlockView::from_bytes(&buf[..])
            .ok()
            .expect("buffer must still parse for undo");
        engine
            .undo_block(block_for_undo)
            .expect("undo block must succeed");
        assert_eq!(engine.accumulated_vote[3], 2000);
    }

    #[test]
    fn check_header_rejects_each_inconsistent_field() {
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(3, 4000);
        engine.set_accumulated_vote_for_test(5, 2000);
        let crypto = test_crypto();
        let all = |_| true;

        // Normal block: wrong value; own credits counted (old definition);
        // non-zero deviation field.
        let nt = nt_with_vote(3);
        for (consumed, recorded) in [(2000, 0), (5200, 0), (4200, 1)] {
            let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
            let block = make_block_view(
                3,
                consumed,
                3,
                recorded,
                Some(nt.as_bytes()),
                &mut buf,
                &crypto,
            );
            assert_eq!(
                engine.check_header(&block, true, all),
                Ok(false),
                "{consumed}/{recorded}"
            );
        }
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 4200, 3, 0, Some(nt.as_bytes()), &mut buf, &crypto);
        assert_eq!(engine.check_header(&block, true, all), Ok(true));

        // Deviation block: the creator's own credit counted (AR-005); a wrong
        // first-voted snapshot; a first-voted node that is not the top.
        for (consumed, recorded) in [(3100, 4200), (2100, 5200)] {
            let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
            let block = deviation_with_credits(&mut buf, &crypto, consumed, recorded);
            assert_eq!(
                engine.check_header(&block, true, all),
                Ok(false),
                "{consumed}/{recorded}"
            );
        }
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(7, 0, 5, 2100, None, &mut buf, &crypto);
        assert_eq!(engine.check_header(&block, true, all), Ok(false));
        // ... which a partial projection cannot judge: only seeded values count.
        assert_eq!(engine.check_header(&block, false, |n| n == 5), Ok(true));
        assert_eq!(engine.check_header(&block, false, |_| false), Ok(true));
    }

    #[test]
    fn check_header_skips_unseeded_values_in_a_partial_projection() {
        // A window-anchored pass: the creator is unseeded, so its value is not
        // judged; the normal form still is.
        let engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(4, 9999, 4, 0, None, &mut buf, &crypto);
        assert_eq!(engine.check_header(&block, false, |_| false), Ok(true));
        assert_eq!(engine.check_header(&block, true, |_| false), Ok(false));
        let mut buf2 = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block2 = make_block_view(4, 9999, 4, 1, None, &mut buf2, &crypto);
        assert_eq!(engine.check_header(&block2, false, |_| false), Ok(false));
    }

    #[test]
    fn check_header_reports_interest_overflow_only_where_it_judges() {
        // The creator's interest step overflows: an error where the value is
        // judged, and no evaluation at all for an unseeded node in a partial
        // projection (the `&&` short-circuits before the snapshot).
        let mut engine = TestEngine::new_for_test(test_vote_scale(), TEST_VOTE_INTEREST);
        engine.set_accumulated_vote_for_test(3, u32::MAX - 10);
        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_block_view(3, 0, 3, 0, None, &mut buf, &crypto);
        assert_eq!(
            engine.check_header(&block, true, |_| true),
            Err(VoteEngineError::AccumulatedVoteOverflow)
        );
        assert_eq!(engine.check_header(&block, false, |_| false), Ok(true));
    }

    #[test]
    fn check_header_requires_parameters() {
        let mut slot = core::mem::MaybeUninit::<TestEngine>::uninit();
        let engine = TestEngine::init(&mut slot);
        let crypto = test_crypto();
        let mut buf = [0u8; moonblokz_chain_types::MAX_BLOCK_SIZE];
        let block = make_empty_block(3, &mut buf, &crypto);
        assert_eq!(
            engine.check_header(&block, true, |_| true),
            Err(VoteEngineError::NotParameterized)
        );
    }
}
