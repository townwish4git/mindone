dir_path="./envs"
files=$(ls -1v "$dir_path"/*.json)

start_index=1
end_index=8

args=""
for file in $(echo "$files" | sed -n "$start_index,${end_index}p"); do
  if [ -f "$file" ]; then
    # python3 ../merge_hccl.py "$file"
    args+=" $file"
  fi
done

echo $args
python3 merge_hccl.py $args "${start_index}_to_${end_index}"