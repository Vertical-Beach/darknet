import os
import json

def check(filename):
    jobj = json.load(open(filename))
    assert(len(jobj) == 150)

def process(filename):
    print(filename)
    jobj = json.load(open(filename))
    car_counts = {}
    ped_counts = {}
    # for all objects and class, count the number of frames which bbox size > 1024pix^2
    for frameres in jobj:
        if "Car" in frameres:
            for car in frameres["Car"]:
                objid = car["id"]
                x1, y1, x2, y2 = car["box2d"]
                size = (x2-x1) * (y2-y1)
                if size >= 1024:
                    if objid not in car_counts:
                        car_counts[objid] = 0
                    car_counts[objid] += 1
        if "Pedestrian" in frameres:
            for ped in frameres["Pedestrian"]:
                objid = ped["id"]
                x1, y1, x2, y2 = ped["box2d"]
                size = (x2-x1) * (y2-y1)
                if size >= 1024:
                    if objid not in ped_counts:
                        ped_counts[objid] = 0
                    ped_counts[objid] += 1
    print(car_counts)
    # don't append object which count is less than 3
    new_results = []
    for frameres in jobj:
        new_frameres = {}
        if "Car" in frameres:
            for car in frameres["Car"]:
                objid = car["id"]
                if objid not in car_counts or car_counts[objid] < 3:
                    continue
                if "Car" not in new_frameres:
                    new_frameres["Car"] = []
                new_frameres["Car"].append(car)

        if "Pedestrian" in frameres:
            for ped in frameres["Pedestrian"]:
                objid = ped["id"]
                if objid not in ped_counts or ped_counts[objid] < 3:
                    continue
                if "Pedestrian" not in new_frameres:
                    new_frameres["Pedestrian"] = []
                new_frameres["Pedestrian"].append(ped)
        new_results.append(new_frameres)
    return new_results

def get_json_name(i):
    return f'test_{str(i).zfill(2)}.json'
def main():
    all_results = {}
    for i in range(74):
        before_json_name = get_json_name(i)
        check(before_json_name)
    
    for i in range(74):
        before_json_name = get_json_name(i)
        video_name = before_json_name.replace('.json', '.mp4')
        all_results[video_name] = process(before_json_name)
    
    open('submit.json', 'w').write(json.dumps(all_results))

# process('test_04.json')
main()