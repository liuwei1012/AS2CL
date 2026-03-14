
# PSM
python main.py --anormly_ratio 1.0 --num_epochs 3 --batch_size 64 --mode train --dataset PSM --data_path ./data/PSM/PSM/ --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None

# # test
python main.py --anormly_ratio 1.0 --num_epochs 100   --batch_size 64  --mode test --dataset PSM  --data_path ./data/PSM/PSM/  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test



# # MSL

python main.py --anormly_ratio 1.0 --num_epochs 2 --batch_size 8 --mode train --dataset MSL --data_path ./data/MSL/MSL/ --input_c 55 --lambd 0.01 --lr 1e-4
# # test
python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 16  --mode test --dataset MSL  --data_path ./data/MSL/MSL/  --input_c 55



# # WADI

python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 16 --mode train --dataset WADI --data_path ./data/WADI/WADI/ --input_c 123 --output_c 123 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None

# # test
python main.py --anormly_ratio 0.04 --num_epochs 100   --batch_size 32  --mode test --dataset WADI  --data_path ./data/WADI/WADI/  --input_c 123 --output_c 123 --n_memory 10 --memory_initial False --phase_type test




# PSM test 3
python main.py --anormly_ratio 1.0 --num_epochs 2 --batch_size 8 --mode train --dataset PSM --data_path ./data/PSM/PSM/ --input_c 25 --lambd 0.01 --lr 1e-4

# # test

python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 16  --mode test --dataset PSM  --data_path ./data/PSM/PSM/  --input_c 25