import datetime

import lief
import numpy as np
from ember import PEFeatureExtractor
from scipy.spatial import distance

from libs.build_section import build_section


def find_adv(differences_index, target, index_file, model, id_file, STEP_MATCH, folder, section_n, dir_pe, resume=False,
             OPTIMIZED=False, c=100):
    id_file = str(id_file)
    step_thresh = 1000 * STEP_MATCH
    first_step = True
    threshold = 0.0001
    step = c
    tollerance_thresh = 5
    size_complexity = 0

    if index_file == -1:
        print("not found")
    else:
        mz = 0
        res = []
        file_data = open(dir_pe + "samples/" + str(index_file), "rb").read()
        extractor = PEFeatureExtractor(2)
        pe_extracted = np.array(extractor.feature_vector(file_data), dtype=np.float64)
        lief_binary = lief.PE.parse(list(file_data))
        lief_adv: lief.PE.Binary = lief.PE.parse(list(file_data))

        target_n = np.array(target[0][mz])
        target_score = model.predict(np.array(target_n)[0:-1].reshape(1, -1))

        if resume:
            file_data_c = open(folder + "temp-" + str(id_file) + "-adv.exe", "rb").read()
            lief_binary_c = lief.PE.parse(list(file_data_c))
            content = lief_binary_c.get_section("test0").content
            c = np.unique(np.array(content), return_counts=True)

        to_add = np.zeros(256)
        if resume:
            for c1, e in enumerate(c[0]):
                to_add[c[0][c1]] = c[1][c1]
        ###

        bin_data = file_data
        counts_bin_orig = np.bincount(np.frombuffer(bin_data, dtype=np.uint8), minlength=256)
        counts_bin_orig = np.array(counts_bin_orig, dtype=np.float32)

        counts_bin = counts_bin_orig
        sum = counts_bin.sum()
        normalized = counts_bin / sum

        tollerance = -1

        dist = distance.euclidean(normalized, target_n[0:256])
        index_range = differences_index[0][mz][0][0:-1]
        counter = 0
        result = [[0]]
        start = datetime.datetime.now()
        if resume:
            counter_total = int(((to_add.sum() + step) / step))
        else:
            counter_total = 0

        while dist > threshold and tollerance < tollerance_thresh and counter_total < step_thresh:
            counter_total = counter_total + 1
            counter = counter + 1

            news = np.concatenate([normalized, pe_extracted[256:]])
            res = {"counter_total": counter_total, "dist_flag": dist > threshold,
                   "tollerance_flag": tollerance < tollerance_thresh, "time": (datetime.datetime.now() - start).seconds,
                   "res": None}

            if OPTIMIZED and counter == STEP_MATCH:
                result = model.predict(news.reshape(1, -1))
                counter = 0
            if (not OPTIMIZED and counter == STEP_MATCH) or (counter == 0 and np.argmax(result[0]) == 0):
                counter = 0
                # print("ADVERSARIAL TRY")
                adv = lief.PE.parse(list(file_data))
                to_add_s = build_section(to_add)
                to_add_t = np.array_split(np.array(to_add_s), section_n)
                to_add_s = np.array(list(to_add_t))
                if to_add_s.shape[0] != section_n:
                    print("ERROR")
                for section_i in range(section_n):
                    section = lief.PE.Section("test" + str(section_i))
                    section.content = to_add_s[section_i]
                    section.size = len(section.content)
                    adv.add_section(section)
                    size_complexity = size_complexity + len(section.content)

                # print(str(section.size)+" "+str(adv.get_section("test0").size))  # section's size

                adv.write(folder + "temp-" + id_file + "-adv.exe")

                file_data_temp = open(folder + "temp-" + id_file + "-adv.exe", "rb").read()
                pe_extracted_adv = np.array(extractor.feature_vector(file_data_temp), dtype=np.float64)
                final = model.predict(pe_extracted_adv.reshape(1, -1))
                print("{" + str(counter_total) + "/" + str(step_thresh) + "}" + str(
                    index_range) + "\tDISTANCE FOR " + str(id_file) + ": ", dist, "\t", tollerance, first_step,
                      "\t" + str(int(target_score[0][0] * 100)) + "\t<- " + str(int(final[0][0] * 100)))
                if np.argmax(final[0]) == 0:
                    print("GOOL")
                    # lief_binary = lief.PE.parse(list(file_data))
                    adv.write(folder + "final-" + str(id_file) + "-step-" + str(STEP_MATCH) + "-section-" + str(
                        section_n) + "-adv.exe")
                    res = {"counter_total": counter_total, "dist_flag": dist > threshold,
                           "tollerance_flag": tollerance < tollerance_thresh,
                           "time": (datetime.datetime.now() - start).seconds, "res": final[0][0],
                           "size_complexity": size_complexity}
                    import json
                    with open(folder + str(id_file) + "-step-" + str(STEP_MATCH) + "-section-" + str(
                            section_n) + "-adv.json", 'w',
                              encoding='utf-8') as f:
                        json.dump(str(res), f, ensure_ascii=False, indent=4)

                    return res

            if distance.euclidean(normalized, target_n[0:256]) > dist:
                tollerance = tollerance + 1
            else:
                tollerance = -1
            dist = distance.euclidean(normalized, target_n[0:256])
            if tollerance > 10:
                first_step = False
                tollerance = -1
            if OPTIMIZED:
                print(str(index_range) + "\tDISTANCE FOR " + str(id_file) + ": ", dist, "\t", tollerance, first_step,
                      "\t" + str(int(target_score[0][0] * 100)) + "\t<- " + str(int(result[0][0] * 100)))

            #   else:
            #      print(str(index_range) + "\tDISTANCE FOR " + str(mi) + ": ", dist, "\t", tollerance, first_step,
            # l              "\t" + str(int(target_score[0][0] * 100)) + "\t<- " + str(int(final[0][0] * 100)))
            counts_bin = counts_bin_orig + to_add

            sum = counts_bin.sum()
            normalized = counts_bin / sum
            if not first_step:
                index_range = range(256)
            for index in index_range:
                # print(index, "original " + str(counts_bin[index]), "\t\tnorm " + str(normalized[index]), "-->\t", target[index])
                if target_n[index] > normalized[index]:
                    to_add[index] = to_add[index] + step
                else:
                    to_add[index] = to_add[index] - step

        print("ok")

    res = {"counter_total": counter_total, "dist_flag": dist > threshold,
           "tollerance_flag": tollerance < tollerance_thresh,
           "time": (datetime.datetime.now() - start).seconds, "res": None, "size_complexity": size_complexity}

    import json
    with open(folder + str(id_file) + "-step-" + str(STEP_MATCH) + "-section-" + str(section_n) + "-adv.json", 'w',
              encoding='utf-8') as f:
        json.dump(str(res), f, ensure_ascii=False, indent=4)

    return res
