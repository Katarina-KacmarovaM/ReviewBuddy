"""LM usage tracking using DSPy lm.history (cost read directly from history)."""

_lm_registry: list = []  # list of (lm, pricing_model, display_name)


def register_lm(lm, pricing_model: str, display_name: str = None) -> None:
    """Register an LM for usage tracking. Requires dspy.configure(track_usage=True)."""
    name = display_name or getattr(lm, "model", pricing_model).split("/")[-1].replace("-fiit", "")
    _lm_registry.append((lm, pricing_model, name))


def snapshot_per_model() -> dict:
    """Return current cumulative usage per model: {name: (prompt, completion, cost)}."""
    snapshot = {}
    for lm, _, name in _lm_registry:
        p, c, cost = 0, 0, 0.0
        for h in (lm.history or []):
            u = h.get("usage") or {}
            p += u.get("prompt_tokens", 0)
            c += u.get("completion_tokens", 0)
            try:
                cost += float(h.get("cost") or 0.0)
            except (TypeError, ValueError):
                pass
        snapshot[name] = (p, c, cost)
    return snapshot


def get_lm_usage():
    """Return (prompt_tokens, completion_tokens, estimated_cost) across all registered LMs."""
    snap = snapshot_per_model()
    total_p = sum(v[0] for v in snap.values())
    total_c = sum(v[1] for v in snap.values())
    total_cost = sum(v[2] for v in snap.values())
    return total_p, total_c, total_cost


def print_cost_summary(
    label: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    elapsed_sec: float,
    n_examples: int = None,
    before_snapshot: dict = None,
    after_snapshot: dict = None,
) -> float:
    """Print cost and speed summary with per-model delta breakdown. Returns cost."""
    mins = elapsed_sec / 60
    print(f"\n{'='*60}")
    print(f"COST & SPEED: {label}")
    print(f"{'='*60}")

    if before_snapshot and after_snapshot:
        for name in after_snapshot:
            p_before, c_before, cost_before = before_snapshot.get(name, (0, 0, 0.0))
            p_after, c_after, cost_after = after_snapshot[name]
            dp = p_after - p_before
            dc = c_after - c_before
            dcost = cost_after - cost_before
            if dp > 0 or dc > 0:
                print(f"  [{name}]  prompt={dp:,}  completion={dc:,}  ${dcost:.4f}")
        print(f"  ---")

    print(f"  Prompt tokens:     {prompt_tokens:,}")
    print(f"  Completion tokens: {completion_tokens:,}")
    print(f"  Total tokens:      {prompt_tokens + completion_tokens:,}")
    print(f"  Cost:              ${cost:.4f}")
    print(f"  Wall time:         {mins:.1f} min ({elapsed_sec:.0f} sec)")
    if n_examples and n_examples > 0:
        print(f"  Per example:       {elapsed_sec/n_examples:.1f} sec, ${cost/n_examples:.4f}")
    print(f"{'='*60}\n")
    return cost
