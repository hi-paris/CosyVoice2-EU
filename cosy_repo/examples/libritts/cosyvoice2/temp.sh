# # 1) Rebuild lists that training actually reads
# cosy="/tsi/hi-paris/tts/Luka/data/cosyvoice2_combined_3000h_mix"

# find "$cosy/train/parquet" -maxdepth 1 -type f -name 'parquet_*.tar' | sort > "$cosy/train.data.list"
# find "$cosy/dev/parquet"   -maxdepth 1 -type f -name 'parquet_*.tar' | sort > "$cosy/dev.data.list"

# # 2) Sanity check: every path exists
# while read -r f; do [[ -f "$f" ]] || echo "MISSING $f"; done < "$cosy/train.data.list"
# while read -r f; do [[ -f "$f" ]] || echo "MISSING $f"; done < "$cosy/dev.data.list"





# for below:

cosy="/tsi/hi-paris/tts/Luka/data/cosyvoice2_combined_3000h_mix"

for split in train dev test; do
  dir="$cosy/$split/parquet"
  list="$dir/data.list"
  [[ -d "$dir" ]] || { echo "Skip $split (no $dir)"; continue; }

  echo "Rebuilding $list"
  tmp="$list.tmp"
  find "$dir" -maxdepth 1 -type f -name 'parquet_*.tar' -print0 \
    | xargs -0 realpath 2>/dev/null \
    | sort > "$tmp"
  mv -f "$tmp" "$list"

  # sanity summary
  total=$(wc -l < "$list" || echo 0)
  missing=$(awk '{print $0}' "$list" | while read -r f; do [[ -f "$f" ]] || echo "$f"; done | wc -l)
  echo "  -> $total entries, $missing missing"
done

