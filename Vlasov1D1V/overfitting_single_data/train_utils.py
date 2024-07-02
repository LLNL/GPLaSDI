import numpy as np
import sys

class ScoreMeter:
    def __init__(self, score_names):
        self.score_names = score_names
        self.scores_sum = np.zeros(len(score_names))
        self.count = 0

    def update(self, scores, n=1):
        self.scores_sum += scores * n
        self.count += n

    def reset(self):
        self.__init__(self.score_names)

    def stats(self):
        scores_avg = self.scores_sum / self.count
        return scores_avg

    def stats_dict(self):
        scores_avg = self.stats()
        stats_dict = {}
        for i, score_name in enumerate(self.score_names):
            stats_dict[score_name] = scores_avg[i]
        return stats_dict

    def stats_string(self, fmt=".2e"):
        scores_avg = self.stats()
        stats_str = ""
        for i, score_name in enumerate(self.score_names):
            stats_str += f"{score_name}: {scores_avg[i]:{fmt}} | "
        return stats_str[:-3]


class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class CSVWriter:
    def __init__(self, path, header):
        self.path = path
        self.header = header
        self.file = open(path, "w")
        self.file.write(header + "\n")
        self.file.flush()

    def write(self, line):
        self.file.write(line + "\n")
        self.file.flush()

    def close(self):
        self.file.close()