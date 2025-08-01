import VisualRelationshipDetection_Dataset as vrd
from Configs import CLIPConfig
from RelationCalculator import Relation_Calculator
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from OvSGTR.datasets.vg import VGDataset
import sys
import os


def get_list_of_objects_from_edges(edges):
    # Extract unique objects from edges
    objects = set()
    for edge in edges:
        objects.add(int(edge[0]))
        objects.add(int(edge[1]))
    return list(objects)

def get_relation_edges_for_objects(edges):
    # background_objects = {
    #     "sky", "clouds", "road", "street", "sidewalk", "ground", "floor",
    #     "building", "wall", "mountain", "hill", "grass", "sand",
    #     "river", "sea", "lake", "snow", "dirt", "ceiling", "curb", "parking lot",
    #     "pavement", "trail", "path", "field", "bridge", "pole", "light pole", "lamp post",
    #     "lamp", "bush", "fence", "barrier", "sidewalk 1", "the wall", "the ground",
    #     "concrete", "railroad tracks", "train tracks", "tracks", "driveway", "crosswalk",
    #     "platform", "floor mat", "stairs", "ceiling fan", "roof", "roof 1", "roof 2",
    #     "skyline", "walls", "fog", "smoke", "shadow", "sun", "moon", "rain", "snow trail",
    #     "seaweed", "sandwich", "ice", "fog lights", "background"
    # } # remove tree,  from the list
    
    relation_edges = {}
    objects = get_list_of_objects_from_edges(edges)
    for i in objects:
        # if object_names[i] in background_objects:
        #     continue
        relation_edges[i] = set()
        first_object = edges[(edges[:, 0] == i)]
        second_object = edges[(edges[:, 1] == i)]
        for j in first_object:
            # if object_names[j[1]] in background_objects:
            #     continue
            relation_edges[i].add(int(j[1]))
        for j in second_object:
            # if object_names[j[0]] in background_objects:
            #     continue
            relation_edges[i].add(int(j[0]))
        if len(relation_edges[i]) == 0:
            del relation_edges[i]
    return relation_edges
def transform_boxes(bboxes, init_image_size ,dest_image_size):
    # Normalize bounding boxes to the image size
    
    bboxes = bboxes.to("cuda")
    bboxes[:, 0] = bboxes[:, 0] * dest_image_size[1] / init_image_size[1]
    bboxes[:, 1] = bboxes[:, 1] * dest_image_size[0] / init_image_size[0]
    bboxes[:, 2] = bboxes[:, 2] * dest_image_size[1] / init_image_size[1]
    bboxes[:, 3] = bboxes[:, 3] * dest_image_size[0] / init_image_size[0]
    
    return bboxes
    
    return bboxes
def process_each_config(config: CLIPConfig, data): 
    layer_results = dict()
    print (f"start thread for config: model={config.model.config._name_or_path}, which_function={config.which_function}")
    n_l = config.num_layers
    for i in range(n_l):
        CLIP_performance_recall = []
        for idx, d in enumerate(data):
            image, target = d
            edges = target['edges'].to(config.device)
            # object_names = target['object_names']
            relation_edges = get_relation_edges_for_objects(edges)
            image_copy = image.copy()
            image_copy = image_copy.resize((config.image_size, config.image_size))
            image_relation_calculator = Relation_Calculator(image_copy, i, config)

            # Normalize bounding boxes to the image size
            bboxes = target['boxes'].to(config.device)
            bboxes = transform_boxes(bboxes, [image.size[1],image.size[0]], (config.image_size,config.image_size))
            
            count_image_recall_objects = 0 
            count_total_relations = 0 
            # image_recall_list = []
            for j in relation_edges.keys():
                attentions_j = dict()
                for k in relation_edges.keys():
                    if j == k:
                        continue
                    attentions_j[k] = image_relation_calculator.get_relation(bboxes[j], bboxes[k])
                n = len(relation_edges[j])
                attentions_j = dict(sorted(attentions_j.items(), key=lambda x: x[1], reverse=True)[:n])
                common_count = len(set(attentions_j.keys()) & relation_edges[j])
                count_image_recall_objects+= common_count
                count_total_relations += n

            CLIP_performance_recall.append(count_image_recall_objects/count_total_relations if count_total_relations > 0 else 0)

            if idx % 1000 == 0:
                print(f"result config: model={config.model_link}, which_function={config.which_function}, layer: {i} for until image {idx} is {sum(CLIP_performance_recall) / len(CLIP_performance_recall)}")
            
        print(f"CLIP performance recall for model={config.model_link}, which_function={config.which_function}, layer={i}: {sum(CLIP_performance_recall) / len(CLIP_performance_recall)}")
        layer_results[i] = sum(CLIP_performance_recall) / len(CLIP_performance_recall)
        
        model_name_safe = config.model_link.replace("/", "_")
        with open(f"result/m_{model_name_safe}-f_{config.which_function}.json", "w") as f:
            json.dump(layer_results, f, indent=4)

    layer_results= dict(sorted(layer_results.items(), key=lambda x: x[0],reverse=True))
    
def process_single_layer (layer, config: CLIPConfig, data):
    print(f"Processing layer {layer} for model {config.model_link} with function {config.which_function}")
    CLIP_performance_recall = []
    for idx, d in enumerate(data):
            image, target = d
            edges = target['edges'].to(config.device)
            # object_names = target['object_names']
            relation_edges = get_relation_edges_for_objects(edges)
            image_copy = image.copy()
            image_copy = image_copy.resize((config.image_size, config.image_size))
            image_relation_calculator = Relation_Calculator(image_copy, layer, config)

            # Normalize bounding boxes to the image size
            bboxes = target['boxes'].to(config.device)
            bboxes = transform_boxes(bboxes, [image.size[1],image.size[0]], (config.image_size,config.image_size))
            
            count_image_recall_objects = 0 
            count_total_relations = 0 
            # image_recall_list = []
            for j in relation_edges.keys():
                attentions_j = dict()
                for k in relation_edges.keys():
                    if j == k:
                        continue
                    attentions_j[k] = image_relation_calculator.get_relation(bboxes[j], bboxes[k])
                n = len(relation_edges[j])
                attentions_j = dict(sorted(attentions_j.items(), key=lambda x: x[1], reverse=True)[:n])
                common_count = len(set(attentions_j.keys()) & relation_edges[j])
                count_image_recall_objects+= common_count
                count_total_relations += n
                
            recall = count_image_recall_objects / count_total_relations if count_total_relations > 0 else -1
            if recall != -1:
                CLIP_performance_recall.append(recall)
            
            with open(f'result/layer{layer}.txt', 'a') as file:
                file.write(f'{recall}\n')
                
            if idx % 200 == 0:
                print(f"result config: model={config.model_link}, which_function={config.which_function}, layer: {layer} for until image {idx} is {sum(CLIP_performance_recall) / len(CLIP_performance_recall)}")
    print(f"CLIP performance recall for model={config.model_link}, which_function={config.which_function}, layer={layer}: {sum(CLIP_performance_recall) / len(CLIP_performance_recall)}")
    
    
    
if __name__ == '__main__':
    image_path = "/data/VG_100K"
    data_path = "/root/aida/vg_data"
    print("Loading dataset...")
    sys.path.append('/root/aida/Test_Relation_in_CLIP')
    image_set = "test"
    data = VGDataset(split=image_set,
                     img_dir=os.path.join(image_path, "VG_100K"), 
                     roidb_file=os.path.join(data_path, "stanford_filtered/VG-SGG.h5"), 
                     dict_file=os.path.join(data_path, "stanford_filtered/VG-SGG-dicts.json"), 
                     image_file=os.path.join(data_path, "stanford_filtered/image_data.json"))
    
    print (f"Dataset loaded with {len(data)} images.")
    
    configs = [
    CLIPConfig(model_link="openai/clip-vit-large-patch14-336", which_function=0, image_size=336, patch_size=14, num_layers=24),
    CLIPConfig(model_link="openai/clip-vit-large-patch14-336", which_function=1, image_size=336, patch_size=14, num_layers=24),
    CLIPConfig(model_link="openai/clip-vit-large-patch14-336", which_function=2, image_size=336, patch_size=14, num_layers=24),
    CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=0, image_size=224, patch_size=16, num_layers=12),
    CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=1, image_size=224, patch_size=16, num_layers=12),
    CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=2, image_size=224, patch_size=16, num_layers=12),
    # CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=4, image_size=224, patch_size=16, num_layers=12),
    # CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=3, image_size=224, patch_size=16, num_layers=12)
    ]

    # process_each_config(configs[4], data)
    print (f"Starting processing for {len(configs)} configurations...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_single_layer, l, configs[4],data) for l in range(12)]
        
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exceptions from the thread
            except Exception as e:
                print(f"Thread raised an exception: {e}")
        







    


