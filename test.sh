export CUDA_VISIBLE_DEVICES=4,5,6,7

# sglang generate --model-path black-forest-labs/FLUX.1-dev \
#   --prompt "A Logo With Bold Large Text: SGL Diffusion" \
#   --save-output
# export SGLANG_ENABLE_SPEC_V2=1
# sglang serve \
#   --model-path stepfun-ai/Step-3.5-Flash \
#   --served-model-name step3p5-flash \
#   --tp-size 4 \
#   --tool-call-parser step3p5 \
#   --reasoning-parser step3p5 \
#   --speculative-algorithm EAGLE \
#   --speculative-num-steps 3 \
#   --speculative-eagle-topk 1 \
#   --speculative-num-draft-tokens 4 \
#   --enable-multi-layer-eagle \
#   --host 0.0.0.0 \
#   --port 8000

python bench_step3p5_mtp.py
