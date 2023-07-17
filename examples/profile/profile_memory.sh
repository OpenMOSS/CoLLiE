#model_name="huggyllama/llama-13b"
model_name="openlm-research/open_llama_3b"
tp_size=4

#for optim in "adam" "adan" "lion" "sophiag"; do
#  echo "Running ${model_name} with ${optim}"
#  torchrun --nproc_per_node ${tp_size} memory_optim.py \
#  --model_name ${model_name} --use_flash 1 --optim ${optim} --tp_size ${tp_size} --pp_size 1
#done

pp_size=1

for peft in "lora" "prefix-tuning" "p-tuning" "prompt-tuning"; do
  echo "Running ${model_name} with ${peft}"
  torchrun --nproc_per_node ${pp_size} memory_peft.py \
  --model_name ${model_name} --use_flash 1 --peft ${peft} --pp_size ${pp_size}
done
