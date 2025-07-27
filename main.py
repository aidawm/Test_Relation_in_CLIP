import VisualRelationshipDetection_Dataset as vrd
from Configs import CLIPConfig
from RelationCalculator import Relation_Calculator
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_relation_edges_for_objects(edges, object_names):
    background_objects = {
        "sky", "clouds", "road", "street", "sidewalk", "ground", "floor",
        "building", "wall", "mountain", "hill", "grass", "sand",
        "river", "sea", "lake", "snow", "dirt", "ceiling", "curb", "parking lot",
        "pavement", "trail", "path", "field", "bridge", "pole", "light pole", "lamp post",
        "lamp", "bush", "fence", "barrier", "sidewalk 1", "the wall", "the ground",
        "concrete", "railroad tracks", "train tracks", "tracks", "driveway", "crosswalk",
        "platform", "floor mat", "stairs", "ceiling fan", "roof", "roof 1", "roof 2",
        "skyline", "walls", "fog", "smoke", "shadow", "sun", "moon", "rain", "snow trail",
        "seaweed", "sandwich", "ice", "fog lights", "background"
    } # remove tree,  from the list
    
    relation_edges = {}
    for i in range(len(object_names)):
        if object_names[i] in background_objects:
            continue
        relation_edges[i] = set()
        first_object = edges[(edges[:, 0] == i)]
        second_object = edges[(edges[:, 1] == i)]
        for j in first_object:
            if object_names[j[1]] in background_objects:
                continue
            relation_edges[i].add(int(j[1]))
        for j in second_object:
            if object_names[j[0]] in background_objects:
                continue
            relation_edges[i].add(int(j[0]))
        if len(relation_edges[i]) == 0:
            del relation_edges[i]
    return relation_edges

def process_each_config(config: CLIPConfig): 
    layer_results = dict()
    print (f"start thread for config: model={config.model.config._name_or_path}, which_function={config.which_function}")
    n_l = config.num_layers
    for i in range(n_l):
        config_vrd = vrd.VRDConfig(normalize_bboxes=True, make_image_square=True, image_size=config.image_size)
        data = vrd.build_vrd(root='data', split='train', config=config_vrd)
        CLIP_performance_recall = []
        for idx, d in enumerate(data):
            if idx == 750:
                break
            image, target = d
            edges = target['edges'].to(config.device)
            object_names = target['object_names']
            relation_edges = get_relation_edges_for_objects(edges, object_names)
            image_relation_calculator = Relation_Calculator(image, i, config)

            # Normalize bounding boxes to the image size
            bboxes = target['normalized_boxes'].to(config.device)
            bboxes *= config.image_size
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

            CLIP_performance_recall.append(count_image_recall_objects/count_total_relations)

            if idx % 50 == 0:
                print(f"result config: model={config.model_link}, which_function={config.which_function}, layer: {i} for until image {idx} is {sum(CLIP_performance_recall) / len(CLIP_performance_recall)}")
            
        print(f"CLIP performance recall for model={config.model_link}, which_function={config.which_function}, layer={i}: {sum(CLIP_performance_recall) / len(CLIP_performance_recall)}")
        layer_results[i] = sum(CLIP_performance_recall) / len(CLIP_performance_recall)
        
        model_name_safe = config.model_link.replace("/", "_")
        with open(f"m_{model_name_safe}-f_{config.which_function}.json", "w") as f:
            json.dump(layer_results, f, indent=4)

    layer_results= dict(sorted(layer_results.items(), key=lambda x: x[0],reverse=True))
    
if __name__ == '__main__':

    
    configs = [
    CLIPConfig(model_link="openai/clip-vit-large-patch14-336", which_function=0, image_size=336, patch_size=14, num_layers=24),
    CLIPConfig(model_link="openai/clip-vit-large-patch14-336", which_function=1, image_size=336, patch_size=14, num_layers=24),
    CLIPConfig(model_link="openai/clip-vit-large-patch14-336", which_function=2, image_size=336, patch_size=14, num_layers=24),
    CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=0, image_size=224, patch_size=16, num_layers=12),
    CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=1, image_size=224, patch_size=16, num_layers=12),
    CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=2, image_size=224, patch_size=16, num_layers=12),
    CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=4, image_size=224, patch_size=16, num_layers=12),
    # CLIPConfig(model_link="openai/clip-vit-base-patch16", which_function=3, image_size=224, patch_size=16, num_layers=12)
    ]

    process_each_config(configs[6])  # Run the first config to ensure the model is loaded


    # with ThreadPoolExecutor(max_workers=6) as executor:
    #     futures = [executor.submit(process_each_config, config) for config in configs]
        
    #     for future in as_completed(futures):
    #         try:
    #             future.result()  # This will raise any exceptions from the thread
    #         except Exception as e:
    #             print(f"Thread raised an exception: {e}")
        







    


