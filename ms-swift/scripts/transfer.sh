# cd /home/ubuntu/jianwen-us-south-2/tulab/enxin/projects/ms-swift/new_data/frames/LLaVA-Video-178K/
# find . -type f > /tmp/all_files.txt

# 分割并并行传输
split -l 1000 /tmp/all_files.txt /tmp/batch_
total_batches=$(ls /tmp/batch_* | wc -l)
total_files=$(wc -l < /tmp/all_files.txt)

echo "总文件数: $total_files, 分为 $total_batches 个批次"

# 使用文件锁来避免竞争条件
echo 0 > /tmp/progress_counter
start_time=$(date +%s)

for batch in /tmp/batch_*; do
    (
        batch_files=$(wc -l < "$batch")
        rsync -a --whole-file --no-compress --files-from="$batch" \
            /home/ubuntu/jianwen-us-south-2/tulab/enxin/projects/ms-swift/new_data/frames/LLaVA-Video-178K/ \
            ~/jianwen-us-south-2/tulab/enxin/dataset/frames/LLaVA-Video-178K/

        # 使用文件锁更新进度
        (
            flock -x 200
            current=$(cat /tmp/progress_counter 2>/dev/null || echo 0)
            echo $((current + batch_files)) > /tmp/progress_counter

            completed_files=$(cat /tmp/progress_counter)
            percent=$((completed_files * 100 / total_files))
            elapsed=$(($(date +%s) - start_time))

            echo "[进度: $percent%] 完成 $completed_files/$total_files 文件 (用时: ${elapsed}s)"
        ) 200>/tmp/progress.lock
    ) &

    (($(jobs -r | wc -l) >= 8)) && wait
done
wait

echo "传输完成! 总计: $total_files 文件"
rm -f /tmp/batch_* /tmp/all_files.txt /tmp/progress_counter /tmp/progress.lock

