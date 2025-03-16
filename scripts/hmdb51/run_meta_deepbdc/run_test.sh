gpuid=8
N_SHOT=5
DATA_ROOT=filelist/hmdb51-molo
YOURPATH=/home/wyll/DeepBDCka3/checkpoints/hmdb51/112tam2 # PATH of your CKPT, e.g., Mine: /home/wyll/TAMT/checkpoints/hmdb51/VideoMAES_meta_deepbdc_5way_5shot_2TAA
cd ../../../

echo "============= meta-test best_model ============="
MODEL_PATH=$YOURPATH/best_model.tar
python test.py --dataset hmdb51 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT  --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 #>> $YOURPATH/bestlog.txt

echo "============= meta-test last_model ============="
MODEL_PATH=$YOURPATH/last_model.tar
python test.py --dataset hmdb51 --data_path $DATA_ROOT --model VideoMAES --method meta_deepbdc --image_size 112 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 #>> $YOURPATH/lastlog.txt