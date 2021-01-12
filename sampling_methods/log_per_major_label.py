from records import TRAIN_RECORDS, TEST_RECORDS
import numpy as np

N = {
  "normal": 5,
  "probe": 7,
  "dos": 8.5,
  "u2r": 7.5,
  "r2l": 12,
}
# records = TRAIN_RECORDS
records = TEST_RECORDS

label_counts = {}
for major_label in records:
  label_count = 0
  for minor_label in records[major_label]:
    label_count += np.log(records[major_label][minor_label] + 1)
  label_counts[major_label] = label_count

total_count = 0
for major_label in records:
  label_count = 0
  for minor_label in records[major_label]:
    count = int((N[major_label] * np.log(records[major_label][minor_label] + 1))/ label_counts[major_label])
    print(
        # major_label,
        # minor_label,
        count,
    )
    label_count += count
    total_count += count
  # print("per label", label_count)

print(total_count)
