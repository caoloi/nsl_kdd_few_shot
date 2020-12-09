from records import TRAIN_RECORDS, TEST_RECORDS

N = 111.4
# records = TRAIN_RECORDS
records = TEST_RECORDS

sum = 0
for major_label in records:
  for minor_label in records[major_label]:
    sum += records[major_label][minor_label]

total_count = 0
for major_label in records:
  for minor_label in records[major_label]:
    count = int(records[major_label][minor_label] / sum * N)
    print(
        # major_label,
        # minor_label,
        count,
    )
    total_count += count

print(total_count)
