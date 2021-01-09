from constants import CONFIG, SAMPLE_NUM_PER_LABEL
import numpy as np
import sys
import datetime
import pytz
import pathlib


def create_summary(results):
  summary = {}

  for result in results:
    for type in result:
      for label in result[type]:
        if label == "accuracy":
          if label not in summary[type]:
            summary[type][label] = []
          summary[type][label].append(result[type][label])
        else:
          for metric in result[type][label]:
            if metric != "support":
              if type not in summary:
                summary[type] = {}
              if label not in summary[type]:
                summary[type][label] = {}
              if metric not in summary[type][label]:
                summary[type][label][metric] = []
              summary[type][label][metric].append(result[type][label][metric])

  return summary


def print_summary(summary, f=sys.stdout):
  for type in summary:
    print(type, file=f)
    for label in summary[type]:
      print("\t" + label, end="", file=f)
      if label == "accuracy":
        mean = np.mean(summary[type][label])
        std = np.std(summary[type][label])
        min = np.min(summary[type][label])
        max = np.max(summary[type][label])
        print(
            "\t\t"
            + "{:.04f}".format(mean) + " ± " + "{:.04f}".format(std)
            + " min: " + "{:.04f}".format(min)
            + " max: " + "{:.04f}".format(max),
            end="",
            file=f,
        )
      else:
        for metric in summary[type][label]:
          mean = np.mean(summary[type][label][metric])
          std = np.std(summary[type][label][metric])
          min = np.min(summary[type][label][metric])
          max = np.max(summary[type][label][metric])
          print(
              "\t\t" + metric + ": "
              + "{:.04f}".format(mean) + " ± " + "{:.04f}".format(std)
              + " min: " + "{:.04f}".format(min)
              + " max: " + "{:.04f}".format(max),
              end="",
              file=f,
          )
      print("", file=f)
    print("", file=f)


def save_summary(summary):
  if CONFIG["save_report"]:
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    acc = np.mean(summary["last_10"]["accuracy"])
    dir = "./summaries/" + now.strftime("%Y%m%d")
    if not pathlib.Path(dir).exists():
      pathlib.Path(dir).mkdir(parents=True)
    file = pathlib.Path(
        dir + "/" + "{:.04f}".format(acc)[2:6] +
        "_" + now.strftime("%Y%m%d_%H%M%S.txt")
    )
    with file.open(mode="w") as f:
      print(now.strftime("%Y%m%d_%H%M%S"), file=f)
      print("Summary", file=f)
      print("CONFIG:", file=f)
      print(CONFIG, file=f)
      print("SAMPLE_NUM_PER_LABEL:", file=f)
      print(SAMPLE_NUM_PER_LABEL, file=f)
      print_summary(summary, f)
