# moonblokz-vote

Standalone `no_std`, no-alloc MoonBlokz vote engine — per-node accumulated-vote registry with FR37 checked forward accumulation, capped anti-capture interest, reversible rollback, balance-block seeding, creator reset, and FR38 next-eligible-creator selection.

- `no_std`, no-alloc, embassy-free.
- Leaf crate: the only direct dependency is `moonblokz-chain-types` (for `BlockView<'_>` / `TransactionView<'_>` / `ComplexTransactionView<'_>`). **No** direct dependency on `moonblokz-blockchain`, crypto, or radio (chain-types mandates one Schnorr backend feature, so `moonblokz-crypto` appears in the transitive tree).
- Fully deterministic — holds no PRNG. FR38 creator ordering is deterministic (descending vote, ascending node-id tie-break).
- `VoteEngine<const MAX_NODES>` — default per architecture §5: `MAX_NODES = 1000`; SoA `accumulated_vote: [u32; MAX_NODES]` ≈ 4 KB.
- Vote-target selection (scoring) is **out of scope** per ADR-007 — the vote engine consumes the `vote: u32` field already present on each transaction.

Implementation tracked story-by-story in `_bmad-output/implementation-artifacts/sprint-status.yaml`.
