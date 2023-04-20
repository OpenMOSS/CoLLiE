set -x
port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" train.py