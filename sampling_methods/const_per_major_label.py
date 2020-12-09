from records import TRAIN_RECORDS, TEST_RECORDS

N = 100
# records = TRAIN_RECORDS
records = TEST_RECORDS

label_counts = {}
for major_label in records:
  label_count = 0
  for minor_label in records[major_label]:
    if records[major_label][minor_label] > 0:
      label_count += 1
  print(label_count)
  label_counts[major_label] = label_count

total_count = 0
for major_label in records:
  for minor_label in records[major_label]:
    count = int(N / (5 * label_counts[major_label])) if records[major_label][minor_label] > 0 else 0
    print(
        # major_label,
        # minor_label,
        count,
    )
    total_count += count

print(total_count)
