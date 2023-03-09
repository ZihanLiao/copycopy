. ./link.sh || exit 1;
. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

export NCCL_DEBUG=INFO

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0

nj=16

stage=0
stop_stage=0

database=
dir=
checkpoint=
config=
symbol_table=
format=raw
cmvn=true

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
  echo "<unk> 1"  >> ${dict}  # <unk> must be 1
  tool/text2token.py -s 1 -n 1 data/local/train/text | cut -f 2- -d" " \
    | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
    awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
  mkdir -p $dir

  # You have to rm `INIT_FILE` manually when you resume or restart a
  # multi-machine training.
  INIT_FILE=$dir/ddp_init
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  for ((i = 0; i < $num_gpus; ++i));do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python bin/train.py \
        --train_data ${database}/train/data.list \
        --dev_data ${database}/dev/data.list \
        --model_dir $dir \
        --checkpoint $checkpoint \
        --config $config \
        --symbol_table $symbol_table \
        --format $format \
        --ddp.init_method $init_method \
        --ddp.world_size $world_size \
        --ddp.rank $rank \
        --ddp.dist_backend $dist_backend \
        --num_workers 2 \
        $cmvn_opts \
        --pin_memory
  } &
  done
  wait
fi