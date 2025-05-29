python main.py --period eval --num_head 1 --num_layer 6 &
python main.py --period eval --num_head 2 --num_layer 6 &
python main.py --period eval --num_head 4 --num_layer 6 & 
python main.py --period eval --num_head 16 --num_layer 6 &


python main.py --period eval --num_head 8 --num_layer 1 &
python main.py --period eval --num_head 8 --num_layer 2 &
python main.py --period eval --num_head 8 --num_layer 4 &
python main.py --period eval --num_head 8 --num_layer 8 & 
python main.py --period eval --num_head 8 --num_layer 10 &

wait
echo "所有测试任务已完成"