# LLM Stack Blocks Entry

Use this path when the stack order must be parsed by the VLM/LLM from a natural
language instruction.

```bash
cd /home/wxm/code/RobotStackDemo

python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "以红色积木为底，把绿色积木放到红色上面，再把蓝色积木放到绿色上面" \
  --output-dir runtime/llm_stack_blocks_test \
  --model qwen2.5vl:7b-q4_K_M \
  --tcp-offset-tool 0 0 0.15
```

Add `--execute --yes` only when the robot, gripper, RealSense, TF, MoveIt, and
Ollama model are ready.

Generated decision files:

- `initial_order_vlm/vlm_stack_decision_input.json`
- `initial_order_vlm/vlm_stack_decision_raw.json`
- `initial_order_vlm/vlm_stack_decision_validated.json`
- `stack_blocks_decision.json`

Online stack planning is always VLM-first, so no force flag is needed. Code
checks JSON shape, unique detector ids, and required `base_link` geometry. It
does not repair or override stack order with an explicit-color rule parser.
Unknown or repeated ids stop fail-safe.

The closed-loop pick/place, optional second-snapshot XY correction, place
verification, and safety checks remain unchanged. The stack workflow now skips
the close pick snapshot by default; add `--enable-second-pick-snapshot` only
when the first locked observation is not accurate enough for pick XY.
