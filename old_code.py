    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='cartialge lesion',
                        choices=['cartialge lesion', 'bone marrow lesion', 'ill', 'synovitis'])
    parser.add_argument('-p', '--plane', type=str, default='cor',
                        choices=['sagittal', 'cor', 'axial'])
    parser.add_argument('--drop_tags', type=str, default=['DIRTY', 'TRIM', 'STIR', 'FS'])
    parser.add_argument('--types', type=str, default=[ 'PD', 'nothing', 'nothing'])
    parser.add_argument('--prefix_name', type=str, default='tipaFinal')


    parser.add_argument('--load_path', type=str,
                        default='models/model_synovitis_toCAM_synovitis_cor_val_auc_0.6667_train_auc_0.7982_epoch_13.pth')
    args = parser.parse_args()

def pre_tranforms_old(projection, task, drop_researh_tags=['DIRTY'], researh_type=None, conjunction=True, resize=True,
                      dim=(256, 256), crop=True, best_slice=True):
    marking = pd.read_csv('labels/marking_arrays_v2.csv', usecols=['path', 'parent path', 'patient number',
                                                                   'researh type', 'projection', 'hip side',
                                                                   'cartialge lesion',
                                                                   'subchondral cysts', 'bone marrow lesion',
                                                                   'synovitis',
                                                                   'ill', 'best_slice', 'bad_research', 'rectangle'])

    bad_mark = '-'
    proc_marking = marking[marking['bad_research'] != bad_mark]
    proc_marking = proc_marking[proc_marking['projection'] == projection]
    if researh_type:
        if conjunction:
            proc_marking_1 = proc_marking[proc_marking['researh type'].str.contains(researh_type[0], na=False)]
            proc_marking_2 = proc_marking[proc_marking['researh type'].str.contains(researh_type[1], na=False)]
            proc_marking_3 = proc_marking[proc_marking['researh type'].str.contains(researh_type[2], na=False)]
            proc_marking = pd.concat([proc_marking_1, proc_marking_2, proc_marking_3])
            proc_marking = proc_marking.drop_duplicates()
        elif not conjunction:
            proc_marking = proc_marking[
                proc_marking['researh type'].str.contains(researh_type[0], na=False) & proc_marking[
                    'researh type'].str.contains(researh_type[1], na=False)]
    for tag in drop_researh_tags:
        mask = proc_marking['researh type'].str.contains(tag, na=False)
        proc_marking = proc_marking[~mask]

    data_dict = {}
    length = 0

    for i, row in enumerate(proc_marking.iloc):
        label = {}
        data = {}
        path = row['path']
        raw_row = row['rectangle']
        box = re.findall(r"\d+", raw_row)
        integer_map = map(int, box)
        box = list(integer_map)
        try:
            images = np.load(path[2:], allow_pickle=True)
        except ValueError:
            print('not loaded ', path)
        else:
            if len(images.shape) > 2:
                data['label'] = int(row[task])
                # random_label = torch.randint(0, 2, (1,)).item()
                # data['label'] = random_label
                if crop:
                    images = square_crop(images, box)
                if best_slice:
                    slice = int(row['best_slice'])
                    images = slice_crop(images, 0.1, slice)
                if resize:
                    images = resize_cv2(images, dim)
                images = normalize_images_v1(images)
                images = normalize_images_v2(images)

                images = np.stack((images, contrast_stretch(images, n=2), equalize_clahe(images)), axis=1)
                images = normalize_images_v1(images)
                # images = contrast_stretch_2(images)

                data['image'] = images
                data_dict[path] = data
                length += 1
    print('dataset length ', length)

    assert len(list(data_dict.keys())) == length
    return length, data_dict


class MRDataset_old(data.Dataset):

    def __init__(self, task, plane, train, drop_researh_tags=['DIRTY', 'TRIM', 'STIR'],
                 researh_type=None, conjunction=True, resize=True, dim=(256, 256), crop=True, transform=True,
                 best_slice=True, weights=None, ):
        super().__init__()
        self.task = task
        self.plane = plane
        self.train = train
        length, data_dict = pre_tranforms(plane, task, drop_researh_tags, researh_type, conjunction, resize, dim, crop,
                                          best_slice)
        train_keys, test_keys = train_test_split(list(data_dict.keys()), test_size=0.3, shuffle=False)
        self.data = data_dict
        self.length = length
        self.transform = transform
        if self.train:
            self.keys = train_keys
        else:
            self.transform = False
            self.keys = test_keys

        if self.transform:
            self.transforms = A.ReplayCompose([
                # A.Resize(256, 256),
                # A.RandomCrop(234, 234),
                # A.Resize(256,256),
                A.Rotate(limit=5),
                A.HorizontalFlip()
            ])

            self.transforms_random = A.Compose([
                # A.Resize(256, 256),
                # A.RandomCrop(234, 234),
                # A.Resize(256,256),
                A.CLAHE(),
                A.GaussNoise(var_limit=(0.0, 30.0)),
                A.MedianBlur(blur_limit=3),
                A.ElasticTransform(alpha=1, sigma=50),
                # A.InvertImg(),
                # A.Blur(blur_limit=(0, 5)),
                A.RandomGamma(gamma_limit=(60, 140)),
                A.RandomFog(fog_coef_lower=0.0, fog_coef_upper=0.3)
            ])
        # if weights is None:
        #    pos = np.sum(self.labels)
        #    neg = len(self.labels) - pos
        #    self.weights = torch.FloatTensor([1, neg / pos])
        # else:
        #    self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        array = self.data[key]['image']
        label = self.data[key]['label']

        if label == 1:
            label = torch.FloatTensor([[0, 1]])
        elif label == 0:
            label = torch.FloatTensor([[1, 0]])

        if self.transform:
            # array = np.stack((array,)*3, axis=1)
            array = self.make_transforms(array, self.transforms)
            array = self.make_random_transforms(array, self.transforms_random)
        else:
            # array = np.stack((array,)*3, axis=1)
            array = torch.from_numpy(array)
        return array, label, key

    def make_transforms(self, array, transforms):
        random.seed(42)
        i = 1
        data = transforms(image=np.moveaxis(array[0], 0, -1))
        transformed = data['image']
        transformed = np.moveaxis(transformed, -1, 0)
        # transformed = np.moveaxis(transformed, 1, 0)
        array[0] = transformed
        replay = data['replay']
        for image in array[1:]:
            transformed = A.ReplayCompose.replay(replay, image=np.moveaxis(image, 0, -1))['image']
            # transformed = np.moveaxis(transformed, 1, 0)
            array[i] = np.moveaxis(transformed, -1, 0)
            i += 1
        return array

    def make_random_transforms(self, array, random_transforms):

        for i, image in enumerate(array):
            transformed = random_transforms(image=np.moveaxis(image, 0, -1))
            transformed_image = transformed["image"]
            array[i] = np.moveaxis(transformed_image, -1, 0)
        return array 
    
    
    
df = pd.DataFrame(columns = ['file' ,'bone marrow lesion', 'synovitis', 'cartialge lesion',  'subchondral cysts', 'type'])

if not os.path.exists('bml'):
    os.makedirs('bml')
    
font  = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (0,255,0)
lineType               = 2
        
for i, item in enumerate(data_list):    
    
    if ('T2' in item['researh_type']) or ('PD' in item['researh_type']):
                         
        images = item['images']    
        image1 = images[item['best_slice']]
        image1 = np.stack((image1,image1,image1), axis=-1)
        image2 = images[item['best_slice'] + 2]
        name = f'{i:04}'+'.png'
        outname = os.path.join('bml', name)
        #cv2.imwrite(outname, image2)
        #cv2.imwrite(outname, image)
        #figure(figsize=(8, 8), dpi=80)
        #imgplot = plt.imshow(image1)
        #plt.show()
        
        bottomLeftCornerOfText = (10,20)
        cv2.putText(image1, '{} {}'.format(name,item['bone marrow lesion']),
                    bottomLeftCornerOfText,
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imwrite(outname, image1)

        row = {'file':outname ,'bone marrow lesion': item['bone marrow lesion'],   'synovitis':item['synovitis'] ,
               'cartialge lesion':item['cartialge lesion'] ,  'subchondral cysts':item['subchondral cysts'], 'type': item['researh_type']}        
        df2 = pd.DataFrame(row,  index=[i])        
        df = df.append(df2)
        
projection = ['cor']    
researh_types = ['T2', 'PD']
drop_types = ['TRIM']
del item

for element in data_list_p1:
    
        i+=1
        if element['projection'] != projection or not any( 
                [researh_type in element['researh_type'] for researh_type in researh_types]) or any(
            [drop_type in element['researh_type'] for drop_type in drop_types]):
            continue
            
        images = np.load(element['path'], allow_pickle=True)


        images = np.stack((images, images, images), axis=1)
        images = normalize_images_v1(images)
    
   
        image1 = images[element['best_slice']]
    
        name = f'{i:04}'+'_p1.png'
        outname = os.path.join('bml',name)  
        
        #cv2.imwrite(outname, image2)
        #cv2.imwrite(outname, image)
        #figure(figsize=(8, 8), dpi=80)
        #imgplot = plt.imshow(image1)
        #plt.show()
        
        bottomLeftCornerOfText = (10,20)
        cv2.putText(image1, '{} {}'.format(name,element['bone marrow lesion']),
                    bottomLeftCornerOfText,
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        
        cv2.imwrite(outname, image1)
        print('p1 image')
        row = {'file':outname ,'bone marrow lesion': element['bone marrow lesion'],   'synovitis':element['synovitis'] ,
               'cartialge lesion':element['cartialge lesion'] ,  'subchondral cysts':element['subchondral cysts'], 'type': element['researh_type']}  
        
        df2 = pd.DataFrame(row,  index=[i])        
        df = df.append(df2)

        
Dataset = PreprocessedDatasetFull()
data_list_full = Dataset.make_in_memory_dataset(task='cartialge lesion', projection='cor', researh_types=['T1', 'T2', 'PD', 'STIR'],
                                                    drop_types=['TRIM'], crop=False, best_slice=False,
                                                    resize=False)
        
df = pd.DataFrame(columns = ['file' , 'cartialge lesion', 'subchondral cysts', 'bone marrow lesion', 'synovitis',  'type', 'path'])

if not os.path.exists('to_crop/data'):
    os.makedirs('to_crop/data')
if not os.path.exists('to_crop/data2'):
    os.makedirs('to_crop/data2')
    
font  = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (0,255,0)
lineType               = 1
       
for i, item in enumerate(data_list_full):    
    
    images = item['images']    
    image1 = images[item['best_slice']]
    
    image1 = np.moveaxis(image1, 0, -1)
    name = item['path'].split('.')[0]+'.png'
    
    #image1 = np.float32(normalize_images_v2(image1))
    image1 = cv2.UMat(image1)
    bottomLeftCornerOfText = (10,20)
    

    cv2.putText(image1, '{} {}'.format(name,item['cartialge lesion']),
                    bottomLeftCornerOfText,
                    font, 
                    fontScale,
                    fontColor,                
                    lineType)
    
    outname =  os.path.join('to_crop', name)
    
    cv2.imwrite(outname, cv2.UMat(image1))

    row = {'file':outname , 'bone marrow lesion': item['bone marrow lesion'],   'synovitis':item['synovitis'] ,
               'cartialge lesion':item['cartialge lesion'] ,  'subchondral cysts':item['subchondral cysts'],  'type': item['researh_type'], 'path': item['path']}       
    
    df2 = pd.DataFrame(row,  index=[i])        
    df = df.append(df2)    
    
df.to_csv('data.csv', index=False)


#MAKE PNG DATASET
from PIL import Image, ImageDraw
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

df = pd.DataFrame(columns = ['file' , 'bone marrow lesion', 'type', 'path'])


if not os.path.exists('bml'):
    os.makedirs('bml')
    
font  = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (0,255,0)
lineType               = 2
bottomLeftCornerOfText = (10,20)
#font = ImageFont.truetype("sans-serif.ttf", 12)

for i, item in enumerate(data_list):    
    
    if ('FS' in item['researh_type']) or ('STIR' in item['researh_type']) or ('stir' in item['researh_type']) or ('fs' in item['researh_type']):
            
        path = item['path']
        
        try:
            images = np.load(path, allow_pickle=True)
            
        except ValueError:
            print('not loaded ', path)
            
            
        image1 = images[item['best_slice']]
        
        image1 = np.stack((image1, image1, image1), axis=-1)
        #image1 = np.moveaxis(image1,0, -1)
    
        #image2 = images[item['best_slice'] - 1]
        #image2 = np.moveaxis(image2,0, -1)
        
        image1 =  Image.fromarray(np.uint8(image1))
        #image2 =  Image.fromarray(np.uint8(image2))
     
    
        name = f'{i:04}'+'.png'
        #name2 = f'{i:04}'+'_2.png'

        outname = os.path.join('bml', name)
        #outname2 = os.path.join('bml', name2)

        
       
        draw = ImageDraw.Draw(image1)
        draw.text(bottomLeftCornerOfText,'{} {}'.format(name,item['bone marrow lesion']),(0,255,0))
        
     
        #draw = ImageDraw.Draw(image2)
        #draw.text(bottomLeftCornerOfText,'{} {}'.format(name,item['bone marrow lesion']),(0,255,0))
        
      
      
        image1.save(outname, "PNG")   
        #image2.save(outname2, "PNG")

        row = {'file':outname ,'bone marrow lesion': item['bone marrow lesion'], 'type': item['researh_type'], 'path': item['path']}        
        df2 = pd.DataFrame(row,  index=[i])        
        df = df.append(df2)
        
df.to_csv('data.csv', index=False)




types_labels = {'./studies/630da13d-86d1-4702-9920-0e3f21305c1f/t1 sag Lumbar_FIL': 'SpC',
'./studies/630da13d-86d1-4702-9920-0e3f21305c1f/T2 sag STIR lumbar_FIL': 'SpC',
'./studies/630da13d-86d1-4702-9920-0e3f21305c1f/t2 tra Lumbar_FIL': 'BA',
'./studies/40d53877-a652-4419-a745-177baa2b70ed/t2_tse_tra_384': 'BA',
'./studies/40d53877-a652-4419-a745-177baa2b70ed/t1_se_cor_p2_384': 'SC',
'./studies/40d53877-a652-4419-a745-177baa2b70ed/t2_pdFS_384_cor': 'SC',
'./studies/40d53877-a652-4419-a745-177baa2b70ed/t2_tse_ax_384_fs': 'SA',
'./studies/40d53877-a652-4419-a745-177baa2b70ed/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/dcf02fba-8b39-4179-b324-41def3e797e8/t2_tse_tra_384': "BA",
'./studies/dcf02fba-8b39-4179-b324-41def3e797e8/t1_se_cor_p2_384': 'SC',
'./studies/dcf02fba-8b39-4179-b324-41def3e797e8/t2_pdFS_384_cor': 'SC',
'./studies/dcf02fba-8b39-4179-b324-41def3e797e8/t2_tse_ax_384_fs': 'SA',
'./studies/dcf02fba-8b39-4179-b324-41def3e797e8/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/5c654c0d-48ec-44a1-aaac-2477b29377ee/t2_tse_tra_384': 'BA',
'./studies/5c654c0d-48ec-44a1-aaac-2477b29377ee/t1_se_cor_p2_384': 'SC',
'./studies/5c654c0d-48ec-44a1-aaac-2477b29377ee/t2_pdFS_384_cor': 'SC',
'./studies/5c654c0d-48ec-44a1-aaac-2477b29377ee/t2_tse_ax_384_fs': "S",
'./studies/5c654c0d-48ec-44a1-aaac-2477b29377ee/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/c6897783-ff80-459e-b1e7-3f3d8da6f894/t2_tse_cor_384': 'SpC',
'./studies/c6897783-ff80-459e-b1e7-3f3d8da6f894/t2_spc_stir_cor_p3_iso_320': 'SpC',
'./studies/c6897783-ff80-459e-b1e7-3f3d8da6f894/t1_tse_cor_384': 'SpC',
'./studies/c6897783-ff80-459e-b1e7-3f3d8da6f894/t2_tse_cor_384 fs': 'SpC',
'./studies/a4677f76-2693-4e05-bc2f-8c95d1528487/t2_tse_tra_384': 'BA',
'./studies/a4677f76-2693-4e05-bc2f-8c95d1528487/t1_se_cor_p2_384': 'SC',
'./studies/a4677f76-2693-4e05-bc2f-8c95d1528487/t2_pdFS_384_cor': 'SC',
'./studies/a4677f76-2693-4e05-bc2f-8c95d1528487/t2_tse_ax_384_fs': 'SA',
'./studies/a4677f76-2693-4e05-bc2f-8c95d1528487/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/31b05cef-1c18-405a-b44b-b8468f36d491/t2_tse_tra_384': 'BA',
'./studies/31b05cef-1c18-405a-b44b-b8468f36d491/t1_se_cor_p2_384': 'SC',
'./studies/31b05cef-1c18-405a-b44b-b8468f36d491/t2_pdFS_384_cor': 'SC',
'./studies/31b05cef-1c18-405a-b44b-b8468f36d491/t2_tse_ax_384_fs': 'SA',
'./studies/31b05cef-1c18-405a-b44b-b8468f36d491/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/aeb81e7c-6556-407f-a2c7-08527c59e8d1/t2_tse_tra_384': 'BA',
'./studies/aeb81e7c-6556-407f-a2c7-08527c59e8d1/t1_se_cor_p2_384': 'SC',
'./studies/aeb81e7c-6556-407f-a2c7-08527c59e8d1/t2_pdFS_384_cor': 'SC',
'./studies/aeb81e7c-6556-407f-a2c7-08527c59e8d1/t2_tse_ax_384_fs': 'Q',
'./studies/aeb81e7c-6556-407f-a2c7-08527c59e8d1/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/aeb81e7c-6556-407f-a2c7-08527c59e8d1/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/843acdf3-1e29-4373-b0d4-3e41d861e932/t2_tse_tra_384': 'BA',
'./studies/843acdf3-1e29-4373-b0d4-3e41d861e932/t1_se_cor_p2_384': 'SC',
'./studies/843acdf3-1e29-4373-b0d4-3e41d861e932/t2_pdFS_384_cor': 'SC',
'./studies/843acdf3-1e29-4373-b0d4-3e41d861e932/t2_tse_ax_384_fs': 'S',
'./studies/843acdf3-1e29-4373-b0d4-3e41d861e932/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/e67ac369-51db-46d4-8633-b13862caec6b/t2_tse_tra_384': 'BA',
'./studies/e67ac369-51db-46d4-8633-b13862caec6b/t1_se_cor_p2_384': 'SC',
'./studies/e67ac369-51db-46d4-8633-b13862caec6b/t2_pdFS_384_cor': 'SC',
'./studies/e67ac369-51db-46d4-8633-b13862caec6b/t2_tse_ax_384_fs': 'S',
'./studies/e67ac369-51db-46d4-8633-b13862caec6b/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/e67ac369-51db-46d4-8633-b13862caec6b/t2_tse_ax_384_fs': 'S',
'./studies/e67ac369-51db-46d4-8633-b13862caec6b/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/517cbc5d-6f55-482a-ac52-6e59174c71ec/t2_tse_tra_384': 'BA',
'./studies/517cbc5d-6f55-482a-ac52-6e59174c71ec/t1_se_cor_p2_384': 'K',
'./studies/517cbc5d-6f55-482a-ac52-6e59174c71ec/t2_pdFS_384_cor': 'SC',
'./studies/517cbc5d-6f55-482a-ac52-6e59174c71ec/t2_tse_ax_384_fs': 'S',
'./studies/517cbc5d-6f55-482a-ac52-6e59174c71ec/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/4851914d-2f5c-4042-8ecc-2177b6c1b177/t2_tse_tra_384': 'BA',
'./studies/4851914d-2f5c-4042-8ecc-2177b6c1b177/t1_se_cor_p2_384': 'SC',
'./studies/4851914d-2f5c-4042-8ecc-2177b6c1b177/t2_pdFS_384_cor': 'SC',
'./studies/4851914d-2f5c-4042-8ecc-2177b6c1b177/t2_tse_ax_384_fs': 'SA',
'./studies/4851914d-2f5c-4042-8ecc-2177b6c1b177/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/81cdebca-5e93-4e02-ab3e-315f3c4c5922/t2_tse_tra_384': 'BA',
'./studies/81cdebca-5e93-4e02-ab3e-315f3c4c5922/t1_se_cor_p2_384': 'SC',
'./studies/81cdebca-5e93-4e02-ab3e-315f3c4c5922/t2_pdFS_384_cor': 'SC',
'./studies/81cdebca-5e93-4e02-ab3e-315f3c4c5922/t2_tse_ax_384_fs': 'SA',
'./studies/81cdebca-5e93-4e02-ab3e-315f3c4c5922/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/49f036fe-978a-478e-8394-3a18246b6324/t2_tse_tra_384': 'BA',
'./studies/49f036fe-978a-478e-8394-3a18246b6324/t1_se_cor_p2_384': 'SC',
'./studies/49f036fe-978a-478e-8394-3a18246b6324/t2_pdFS_384_cor': 'SC',
'./studies/49f036fe-978a-478e-8394-3a18246b6324/t2_tse_ax_384_fs': "S",
'./studies/49f036fe-978a-478e-8394-3a18246b6324/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/834443e9-c126-4eb1-87d7-cb501bc02c83/t2_tse_tra_384': 'BA',
'./studies/834443e9-c126-4eb1-87d7-cb501bc02c83/t1_se_cor_p2_384': 'SC',
'./studies/834443e9-c126-4eb1-87d7-cb501bc02c83/t2_pdFS_384_cor': 'SC',
'./studies/834443e9-c126-4eb1-87d7-cb501bc02c83/t2_tse_ax_384_fs': 'S',
'./studies/834443e9-c126-4eb1-87d7-cb501bc02c83/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/a87649a2-c8e5-4448-9165-4f4f3138276e/t2_tse_tra_384': 'BA',
'./studies/a87649a2-c8e5-4448-9165-4f4f3138276e/t1_se_cor_p2_384': 'SC',
'./studies/a87649a2-c8e5-4448-9165-4f4f3138276e/t2_pdFS_384_cor': 'SC',
'./studies/a87649a2-c8e5-4448-9165-4f4f3138276e/t2_tse_ax_384_fs': 'SA',
'./studies/a87649a2-c8e5-4448-9165-4f4f3138276e/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/8c8bac38-b5a6-40e5-af64-dc01eb6bc609/t2_tse_tra_384': 'BA',
'./studies/8c8bac38-b5a6-40e5-af64-dc01eb6bc609/t1_se_cor_p2_384': 'SC',
'./studies/8c8bac38-b5a6-40e5-af64-dc01eb6bc609/t2_pdFS_384_cor': 'SC',
'./studies/8c8bac38-b5a6-40e5-af64-dc01eb6bc609/t2_tse_ax_384_fs': 'BA',
'./studies/8c8bac38-b5a6-40e5-af64-dc01eb6bc609/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/8c8bac38-b5a6-40e5-af64-dc01eb6bc609/t2_tse_sag_384': 'SpS',
'./studies/f31047dc-46bb-4254-a7bf-b849da7178fd/t2_tse_tra_384': 'BA',
'./studies/f31047dc-46bb-4254-a7bf-b849da7178fd/t1_se_cor_p2_384': 'SC',
'./studies/f31047dc-46bb-4254-a7bf-b849da7178fd/t2_pdFS_384_cor': 'Q',
'./studies/f31047dc-46bb-4254-a7bf-b849da7178fd/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/f31047dc-46bb-4254-a7bf-b849da7178fd/t2_tse_ax_384_fs': 'SA',
'./studies/5458dc1d-5a24-4576-b130-a0987ca4b6da/t2_tse_tra_384': 'BA',
'./studies/5458dc1d-5a24-4576-b130-a0987ca4b6da/t1_se_cor_p2_384': 'SC',
'./studies/5458dc1d-5a24-4576-b130-a0987ca4b6da/t2_pdFS_384_cor': 'SC',
'./studies/5458dc1d-5a24-4576-b130-a0987ca4b6da/pd_spc_rst_cor_p2_iso_512': 'SC',
'./studies/5458dc1d-5a24-4576-b130-a0987ca4b6da/t2_tse_ax_384_fs': 'BA',
'./studies/536d65be-cde7-4895-998e-4d7adb4681d6/t2_tse_tra_384': 'BA',
'./studies/536d65be-cde7-4895-998e-4d7adb4681d6/t1_se_cor_p2_384': 'SC',
'./studies/536d65be-cde7-4895-998e-4d7adb4681d6/t2_pdFS_384_cor': 'SC',
'./studies/536d65be-cde7-4895-998e-4d7adb4681d6/t2_tse_ax_384_fs': 'SA',
'./studies/536d65be-cde7-4895-998e-4d7adb4681d6/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/a18e3c0d-53bb-44e6-b142-b76d9af1640e/t2_tse_tra_384': 'BA',
'./studies/a18e3c0d-53bb-44e6-b142-b76d9af1640e/t1_se_cor_p2_384': 'SC',
'./studies/a18e3c0d-53bb-44e6-b142-b76d9af1640e/t2_pdFS_384_cor': 'SC', 
'./studies/a18e3c0d-53bb-44e6-b142-b76d9af1640e/t2_tse_ax_384_fs': 'SA',
'./studies/a18e3c0d-53bb-44e6-b142-b76d9af1640e/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/c1dddf0f-9518-4485-9317-77e991ed3b47/t2_tse_tra_384': 'BA',
'./studies/c1dddf0f-9518-4485-9317-77e991ed3b47/t2_pdFS_384_ax': 'REST',
'./studies/c1dddf0f-9518-4485-9317-77e991ed3b47/t2_pdFS_384_cor': 'Q',
'./studies/c1dddf0f-9518-4485-9317-77e991ed3b47/t2_pdFS_384_sag': 'S',
'./studies/c1dddf0f-9518-4485-9317-77e991ed3b47/t1_se_ax_p2_384': 'REST',
'./studies/c1dddf0f-9518-4485-9317-77e991ed3b47/t1_se_ax_p2_384 fs': 'REST',
'./studies/41ed76aa-6ac2-47ca-8ce8-f0e2262e93b6/t2_tse_tra_384': 'BA',
'./studies/41ed76aa-6ac2-47ca-8ce8-f0e2262e93b6/t1_se_cor_p2_384': 'SC',
'./studies/41ed76aa-6ac2-47ca-8ce8-f0e2262e93b6/t2_pdFS_384_cor': 'SC',
'./studies/41ed76aa-6ac2-47ca-8ce8-f0e2262e93b6/t2_tse_ax_384_fs': 'S',
'./studies/41ed76aa-6ac2-47ca-8ce8-f0e2262e93b6/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/566f84d6-fa47-404f-a9ff-9a617a785d3f/t2_tse_tra_384': 'BA',
'./studies/566f84d6-fa47-404f-a9ff-9a617a785d3f/t1_se_cor_p2_384': 'SC',
'./studies/566f84d6-fa47-404f-a9ff-9a617a785d3f/t2_pdFS_384_cor': 'SC',
'./studies/566f84d6-fa47-404f-a9ff-9a617a785d3f/t2_tse_ax_384_fs': 'S',
'./studies/566f84d6-fa47-404f-a9ff-9a617a785d3f/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/05b9aa43-7e4e-4d58-a63d-082bdec57012/t2_tse_tra_384': 'BA',
'./studies/05b9aa43-7e4e-4d58-a63d-082bdec57012/t1_se_cor_p2_384': 'SC',
'./studies/05b9aa43-7e4e-4d58-a63d-082bdec57012/t2_pdFS_384_cor': 'SC',
'./studies/05b9aa43-7e4e-4d58-a63d-082bdec57012/t2_tse_ax_384_fs': 'S',
'./studies/05b9aa43-7e4e-4d58-a63d-082bdec57012/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/05b9aa43-7e4e-4d58-a63d-082bdec57012/t2_pdFS_384_cor_ND': 'SC',
'./studies/c386ba04-2bd4-4106-b617-a9a6e02fa015/t2_tse_tra_384': 'BA',
'./studies/c386ba04-2bd4-4106-b617-a9a6e02fa015/t1_se_cor_p2_384': 'SC',
'./studies/c386ba04-2bd4-4106-b617-a9a6e02fa015/t2_pdFS_384_cor': 'SC',                
'./studies/c386ba04-2bd4-4106-b617-a9a6e02fa015/t2_tse_ax_384_fs': 'SA',
'./studies/c386ba04-2bd4-4106-b617-a9a6e02fa015/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/51a205ce-6a4e-4f28-9302-653d9af5b2fd/t2_tse_tra_384': 'BA',
'./studies/51a205ce-6a4e-4f28-9302-653d9af5b2fd/t1_se_cor_p2_384': 'K',
'./studies/51a205ce-6a4e-4f28-9302-653d9af5b2fd/t2_pdFS_384_cor': 'SC',
'./studies/51a205ce-6a4e-4f28-9302-653d9af5b2fd/t2_tse_ax_384_fs': 'S',
'./studies/51a205ce-6a4e-4f28-9302-653d9af5b2fd/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/252821c7-2b6d-4ade-8ece-2945908b65f2/t2_tse_tra_384': 'BA',
'./studies/252821c7-2b6d-4ade-8ece-2945908b65f2/t1_se_cor_p2_384': 'SC',
'./studies/252821c7-2b6d-4ade-8ece-2945908b65f2/t2_pdFS_384_cor': 'SC',
'./studies/252821c7-2b6d-4ade-8ece-2945908b65f2/t2_tse_ax_384_fs': 'S',
'./studies/252821c7-2b6d-4ade-8ece-2945908b65f2/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/5f9523a5-1dee-4937-82c1-52ef21208905/t2_tse_tra_384':'BA',
'./studies/5f9523a5-1dee-4937-82c1-52ef21208905/t1_se_cor_p2_384': 'SC',
'./studies/5f9523a5-1dee-4937-82c1-52ef21208905/t2_pdFS_384_cor': 'SC',
'./studies/5f9523a5-1dee-4937-82c1-52ef21208905/t2_tse_ax_384_fs': 'SA',
'./studies/5f9523a5-1dee-4937-82c1-52ef21208905/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/5f9523a5-1dee-4937-82c1-52ef21208905/t2_tse_tra_384 fs': 'BA',
'./studies/5f9523a5-1dee-4937-82c1-52ef21208905/t2_tse_sag_384_fs': 'S',
'./studies/86a25b4c-2bcd-4f4c-ae4d-cb69706c8fab/t2_tse_tra_384': 'BA',
'./studies/86a25b4c-2bcd-4f4c-ae4d-cb69706c8fab/t1_se_cor_p2_384': 'SC',
'./studies/86a25b4c-2bcd-4f4c-ae4d-cb69706c8fab/t2_pdFS_384_cor':'SC',
'./studies/86a25b4c-2bcd-4f4c-ae4d-cb69706c8fab/t2_tse_ax_384_fs': 'SA',
'./studies/86a25b4c-2bcd-4f4c-ae4d-cb69706c8fab/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/54a686ea-6eff-4408-80a8-9f0375105e27/t2_tse_tra_384': 'BA',
'./studies/54a686ea-6eff-4408-80a8-9f0375105e27/t1_se_cor_p2_384': 'SC',
'./studies/54a686ea-6eff-4408-80a8-9f0375105e27/t2_pdFS_384_cor': 'SC',
'./studies/54a686ea-6eff-4408-80a8-9f0375105e27/t2_tse_ax_384_fs': 'SA',
'./studies/54a686ea-6eff-4408-80a8-9f0375105e27/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/33ed219a-b345-47dd-a414-d2dd6ce56600/t2_tse_tra_384': 'BA',
'./studies/33ed219a-b345-47dd-a414-d2dd6ce56600/t1_se_cor_p2_384': 'SC',
'./studies/33ed219a-b345-47dd-a414-d2dd6ce56600/t2_pdFS_384_cor': 'SC',
'./studies/33ed219a-b345-47dd-a414-d2dd6ce56600/t2_tse_ax_384_fs': 'SA',
'./studies/33ed219a-b345-47dd-a414-d2dd6ce56600/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/536c6a46-2037-4039-af6b-58ff9a48114e/t2_tse_tra_384': 'BA',
'./studies/536c6a46-2037-4039-af6b-58ff9a48114e/t1_se_cor_p2_384': 'SC',
'./studies/536c6a46-2037-4039-af6b-58ff9a48114e/t2_pdFS_384_cor': 'SC',
'./studies/536c6a46-2037-4039-af6b-58ff9a48114e/t2_tse_sag_384_fs': 'S',
'./studies/536c6a46-2037-4039-af6b-58ff9a48114e/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/534a9969-654b-4dc4-9e10-505e3e67dd92/t2_tse_tra_384': 'BA',
'./studies/534a9969-654b-4dc4-9e10-505e3e67dd92/t1_se_cor_p2_384': 'SC',
'./studies/534a9969-654b-4dc4-9e10-505e3e67dd92/t2_pdFS_384_cor': 'SC',
'./studies/534a9969-654b-4dc4-9e10-505e3e67dd92/t2_tse_ax_384_fs': 'SA',
'./studies/534a9969-654b-4dc4-9e10-505e3e67dd92/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/534a9969-654b-4dc4-9e10-505e3e67dd92/t2_tse_tra_384  fs': 'BA',
'./studies/afbe7240-21d9-4211-8a92-40cfa6abb9e1/t2_tse_tra_384':'BA',
'./studies/afbe7240-21d9-4211-8a92-40cfa6abb9e1/t1_se_cor_p2_384': 'SC',
'./studies/afbe7240-21d9-4211-8a92-40cfa6abb9e1/t2_pdFS_384_cor':'SC',
'./studies/afbe7240-21d9-4211-8a92-40cfa6abb9e1/t2_tse_ax_384_fs': 'S',
'./studies/afbe7240-21d9-4211-8a92-40cfa6abb9e1/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/17c94843-6b33-46fd-a6f9-cf90260517b2/t2_tse_tra_384':'BA',
'./studies/17c94843-6b33-46fd-a6f9-cf90260517b2/t1_se_cor_p2_384':'SC',
'./studies/17c94843-6b33-46fd-a6f9-cf90260517b2/t2_pdFS_384_cor':'SC',
'./studies/17c94843-6b33-46fd-a6f9-cf90260517b2/t2_tse_ax_384_fs': 'S',
'./studies/17c94843-6b33-46fd-a6f9-cf90260517b2/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/17c94843-6b33-46fd-a6f9-cf90260517b2/t2_tse_tra_384 fs': 'BA',
'./studies/2effe877-5b02-44c5-834b-49037aa358c0/t2_tse_tra_384':'BA',
'./studies/2effe877-5b02-44c5-834b-49037aa358c0/t1_se_cor_p2_384':'SC',
'./studies/2effe877-5b02-44c5-834b-49037aa358c0/t2_pdFS_384_cor':'SC',
'./studies/2effe877-5b02-44c5-834b-49037aa358c0/t2_tse_ax_384_fs':'SA',
'./studies/2effe877-5b02-44c5-834b-49037aa358c0/pd_spc_rst_cor_p2_iso_512':'BC',
'./studies/2cd37e46-74c6-46a7-9aa8-034c9c9aa0cf/t2_tse_tra_384':'BA',
'./studies/2cd37e46-74c6-46a7-9aa8-034c9c9aa0cf/t1_se_cor_p2_384':'SC',
'./studies/2cd37e46-74c6-46a7-9aa8-034c9c9aa0cf/t2_pdFS_384_cor':'SC',
'./studies/2cd37e46-74c6-46a7-9aa8-034c9c9aa0cf/t2_tse_sag_384_fs':'S',
'./studies/2cd37e46-74c6-46a7-9aa8-034c9c9aa0cf/pd_spc_rst_cor_p2_iso_512':'BC',
'./studies/2cd37e46-74c6-46a7-9aa8-034c9c9aa0cf/t2_tse_sag_384':'SpS',
'./studies/a44d6a18-df7b-4d40-a642-f20c00c09972/t2_tse_tra_384':'BA',
'./studies/a44d6a18-df7b-4d40-a642-f20c00c09972/t1_se_cor_p2_384':'SC',
'./studies/a44d6a18-df7b-4d40-a642-f20c00c09972/t2_pdFS_384_cor':'SC',
'./studies/a44d6a18-df7b-4d40-a642-f20c00c09972/t2_tse_ax_384_fs': 'SA',
'./studies/a44d6a18-df7b-4d40-a642-f20c00c09972/pd_spc_rst_cor_p2_iso_512':'BC',
'./studies/598e8f4d-cd57-4bbb-afd4-75f8f62f716b/t2_tse_sag_384': 'S',
'./studies/598e8f4d-cd57-4bbb-afd4-75f8f62f716b/t2_tse_cor_384': 'SpC',
'./studies/598e8f4d-cd57-4bbb-afd4-75f8f62f716b/t2_tse_cor_384 fs':'SpC',
'./studies/ed84cf7b-0b46-4f29-80ba-556362752e9d/t2_tse_tra_384': 'BA',
'./studies/ed84cf7b-0b46-4f29-80ba-556362752e9d/t1_se_cor_p2_384': 'SC',
'./studies/ed84cf7b-0b46-4f29-80ba-556362752e9d/t2_pdFS_384_cor': 'SC',
'./studies/ed84cf7b-0b46-4f29-80ba-556362752e9d/t2_tse_sag_384_fs': 'S',
'./studies/ed84cf7b-0b46-4f29-80ba-556362752e9d/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/2ecf1037-e625-48b2-834b-437d8e7a7447/t1_se_cor_p2_384': 'SC',
'./studies/2ecf1037-e625-48b2-834b-437d8e7a7447/t2_pdFS_384_cor': 'SA',
'./studies/2ecf1037-e625-48b2-834b-437d8e7a7447/t2_tse_ax_384_fs': 'SA',
'./studies/2ecf1037-e625-48b2-834b-437d8e7a7447/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/98d99f15-43af-4700-b512-5106a429f310/t1_se_cor_p2_384':'SC',
'./studies/98d99f15-43af-4700-b512-5106a429f310/t2_pdFS_384_cor':'SC',
'./studies/98d99f15-43af-4700-b512-5106a429f310/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/acfee5b0-4c5e-4d02-a721-6669d0f200ea/t1_se_cor_p2_384': 'SC',
'./studies/acfee5b0-4c5e-4d02-a721-6669d0f200ea/t2_pdFS_384_cor': 'SC',
'./studies/acfee5b0-4c5e-4d02-a721-6669d0f200ea/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/96071a0f-07c5-4d99-a1d0-552236289d16/t1_se_cor_p2_384': 'SC',
'./studies/96071a0f-07c5-4d99-a1d0-552236289d16/t2_pdFS_384_cor': 'SC',
'./studies/96071a0f-07c5-4d99-a1d0-552236289d16/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/f3b22cf8-840c-4978-ab13-2d233a61e0d2/t1_se_cor_p2_384': 'SC',
'./studies/f3b22cf8-840c-4978-ab13-2d233a61e0d2/t2_pdFS_384_cor': 'SC',
'./studies/f3b22cf8-840c-4978-ab13-2d233a61e0d2/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/69e2b986-7fc4-4cb2-8cea-ad483b9daacf/t1_se_cor_p2_384': 'SC',
'./studies/69e2b986-7fc4-4cb2-8cea-ad483b9daacf/t2_pdFS_384_cor':'SC',
'./studies/69e2b986-7fc4-4cb2-8cea-ad483b9daacf/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/165a3e8c-f590-4f73-90c5-e8cfd9a27dc7/t1_se_cor_p2_384': 'SC',
'./studies/165a3e8c-f590-4f73-90c5-e8cfd9a27dc7/t2_pdFS_384_cor': 'SC',
'./studies/165a3e8c-f590-4f73-90c5-e8cfd9a27dc7/pd_spc_rst_cor_p2_iso_512':'BC',
'./studies/a8e52abe-fede-4141-abf1-24bd10e81ff4/t1_se_cor_p2_384': 'SC',
'./studies/a8e52abe-fede-4141-abf1-24bd10e81ff4/t2_pdFS_384_cor': 'SC',
'./studies/a8e52abe-fede-4141-abf1-24bd10e81ff4/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/42258d1f-7f2b-4e95-9734-2af24c285f92/t1_se_cor_p2_384': 'SC',
'./studies/42258d1f-7f2b-4e95-9734-2af24c285f92/t2_pdFS_384_cor':'SC',
'./studies/42258d1f-7f2b-4e95-9734-2af24c285f92/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/51ddb7f7-c9a5-4234-bf9a-86fc9e0b528b/t1_se_cor_p2_384': 'SC',
'./studies/51ddb7f7-c9a5-4234-bf9a-86fc9e0b528b/t2_pdFS_384_cor': 'SC',
'./studies/51ddb7f7-c9a5-4234-bf9a-86fc9e0b528b/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/071255b6-ad06-48f8-ae92-d264d3e30f46/t1_se_cor_p2_384': 'SC',
'./studies/071255b6-ad06-48f8-ae92-d264d3e30f46/t2_pdFS_384_cor': 'SC',
'./studies/071255b6-ad06-48f8-ae92-d264d3e30f46/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/d3ac6386-693c-4d46-b4d3-5598339c2060/t1_se_cor_p2_384': 'SC',
'./studies/d3ac6386-693c-4d46-b4d3-5598339c2060/t2_pdFS_384_cor': 'SpC',
'./studies/d3ac6386-693c-4d46-b4d3-5598339c2060/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/99efaf25-36e4-4c3a-9ee1-292fb3847815/t2_tse_cor_p2': 'BC',
'./studies/99efaf25-36e4-4c3a-9ee1-292fb3847815/t2_tse_cor_p2_fs': 'BC',
'./studies/b1ce799b-046f-47c8-8c01-6b3ef4581dc9/t1_se_cor_p2_384': 'SC',
'./studies/b1ce799b-046f-47c8-8c01-6b3ef4581dc9/t2_pdFS_384_cor': 'SC',
'./studies/b1ce799b-046f-47c8-8c01-6b3ef4581dc9/t2_tse_ax_384_fs': 'SA',
'./studies/b1ce799b-046f-47c8-8c01-6b3ef4581dc9/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/f92c8110-2850-4c46-bb0a-5becec510ae6/t1_se_cor_p2_384': 'SC',
'./studies/f92c8110-2850-4c46-bb0a-5becec510ae6/t2_pdFS_384_cor':'SC',
'./studies/f92c8110-2850-4c46-bb0a-5becec510ae6/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/86d6de24-2c07-4072-b271-86890b06d38a/t2_tse_cor_p2': 'BC',
'./studies/86d6de24-2c07-4072-b271-86890b06d38a/t2_tse_cor_p2_fs': 'BC',
'./studies/1030d870-adf4-4abe-bf8a-300ec45a8637/t1_se_cor_p2_384': 'SC',
'./studies/1030d870-adf4-4abe-bf8a-300ec45a8637/t2_pdFS_384_cor': 'SC',
'./studies/1030d870-adf4-4abe-bf8a-300ec45a8637/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/d6e03826-9ec2-4dad-a024-041c0d6d56a2/t2_tse_cor_p2':'BC',
'./studies/7ad5a1be-029e-4b7d-b57f-bef533e6ab05/t1_se_cor_p2_384': 'SC',
'./studies/7ad5a1be-029e-4b7d-b57f-bef533e6ab05/t2_pdFS_384_cor': 'SC',
'./studies/7ad5a1be-029e-4b7d-b57f-bef533e6ab05/t1_se_cor_p2_384 fs':'SC',
'./studies/cb5b937d-574a-4ba6-a9b2-7d656dc0471e/t1_se_cor_p2_384':'SC',
'./studies/cb5b937d-574a-4ba6-a9b2-7d656dc0471e/t2_pdFS_384_cor': 'SC',
'./studies/cb5b937d-574a-4ba6-a9b2-7d656dc0471e/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/1742b962-1971-42b1-b0cc-d7bf2fc6c0b2/t1_se_cor_p2_384': 'SC',
'./studies/1742b962-1971-42b1-b0cc-d7bf2fc6c0b2/t2_pdFS_384_cor': 'SC',
'./studies/1742b962-1971-42b1-b0cc-d7bf2fc6c0b2/pd_spc_rst_cor_p2_iso_512': 'BC',
'./studies/1742b962-1971-42b1-b0cc-d7bf2fc6c0b2/t2_tse_sag_384':'SpS',
'./studies/5e978e26-7cbd-49c8-8c42-3cc9f341ab1b/t1_se_cor_p2_384':'SC',
'./studies/5e978e26-7cbd-49c8-8c42-3cc9f341ab1b/t2_pdFS_384_cor':'SC',
'./studies/5e978e26-7cbd-49c8-8c42-3cc9f341ab1b/pd_spc_rst_cor_p2_iso_512':'BC',


}

print(len(types_labels.keys()))

fs_labels = {
    
    './studies/776ecec0-3726-4588-8100-25db73fdefed/t2_pdFS_384_cor':'SC', 
    './studies/23aefcd4-9ab7-41bc-adac-f543d9062e68/t2_tse_cor_320_p2_fs':'BC',
    './studies/9335ea2e-5182-4bcf-a0f7-f669bbd8abd1/t2_pdFS_384_cor':'SC', 
    './studies/fae44ee2-2871-4adc-8868-3d63b00572a7/t2_pdFS_384_cor':'SC',     
    './studies/7c17faee-eecc-490b-bfd7-04bf551a1c7c/t2_pdFS_384_cor':'SC', 

    './studies/83a15874-e0f7-401b-a55f-f4837570dc2c/t2_pdFS_384_cor':'BC',

    './studies/89997371-abb3-4f1f-ba19-99f02a6c3488/t2_pdFS_384_cor':'SC', 

    './studies/098633de-a6d4-42c2-b458-f8a4588881df/t2_pdFS_384_cor':'SC', 

    './studies/a008f02e-3beb-4dcb-a0fa-492d570707d0/t2_pdFS_384_cor':'SC', 

    './studies/5a3fb268-0b08-40a8-9d3a-824dde0d0903/t2_pdFS_384_cor':'SC', 

    './studies/916cf871-a4de-4185-be0f-abe65da435f5/t2_pdFS_384_cor':'SC', 

    './studies/515c8faa-1b44-41b8-9617-bc3dc31fa2c4/t2_pdFS_384_cor':'SC', 

    './studies/cb3abce1-0e1d-4abd-b168-554ef32b0074/t2_tse_cor_320_p2_fs':'BC',

    './studies/2197db09-935b-4d32-9f4d-eaaa8e0473a1/t2_pdFS_384_cor':'SC', 

    './studies/711de9fe-8fa8-42e5-92cc-e2225c4d31f4/t2_pdFS_384_cor':'SC', 

    './studies/8174e695-c9f8-4f6f-b15d-1c5eaaa025b9/t2_pdFS_384_cor':'SC', 

    './studies/aeb38e60-cd77-4e29-a874-e480c157c0ef/t2_pdFS_384_cor':'SC',   
    './studies/03e0f178-37e3-454b-abe3-c2e2b2568ee1/t2_pdFS_384_cor':'SC',   
    './studies/acc0ff90-7b44-44f5-892a-fc2c25fe0422/t2_pdFS_384_cor':'SC',   
    './studies/f44bc9b7-a9c0-4ca5-a9d9-1cfcd4142a15/t2_pdFS_384_cor':'SC',   
    './studies/2794fdad-b5e8-4e27-b144-a50ea0a25443/t1_se_cor_p2_384 FS': 'SC',   
    './studies/98d99f15-43af-4700-b512-5106a429f310/t2_pdFS_384_cor':'SC',  
    './studies/acfee5b0-4c5e-4d02-a721-6669d0f200ea/t2_pdFS_384_cor':'SC',
    './studies/96071a0f-07c5-4d99-a1d0-552236289d16/t2_pdFS_384_cor':'SC', 
    './studies/f3b22cf8-840c-4978-ab13-2d233a61e0d2/t2_pdFS_384_cor':'SC',
    './studies/69e2b986-7fc4-4cb2-8cea-ad483b9daacf/t2_pdFS_384_cor':'SC',   
    './studies/165a3e8c-f590-4f73-90c5-e8cfd9a27dc7/t2_pdFS_384_cor':'SC',   
    './studies/a8e52abe-fede-4141-abf1-24bd10e81ff4/t2_pdFS_384_cor':'SC',   
    './studies/42258d1f-7f2b-4e95-9734-2af24c285f92/t2_pdFS_384_cor':'SC',   
    './studies/51ddb7f7-c9a5-4234-bf9a-86fc9e0b528b/t2_pdFS_384_cor':'SC',   
    './studies/071255b6-ad06-48f8-ae92-d264d3e30f46/t2_pdFS_384_cor':'SC',   
    './studies/99efaf25-36e4-4c3a-9ee1-292fb3847815/t2_tse_cor_p2_fs': 'BC', 
    './studies/b1ce799b-046f-47c8-8c01-6b3ef4581dc9/t2_pdFS_384_cor':'SC',   
    './studies/f92c8110-2850-4c46-bb0a-5becec510ae6/t2_pdFS_384_cor':'SC',   
    './studies/86d6de24-2c07-4072-b271-86890b06d38a/t2_tse_cor_p2_fs':'BC', 
    './studies/1030d870-adf4-4abe-bf8a-300ec45a8637/t2_pdFS_384_cor':'SC',   
    './studies/d6e03826-9ec2-4dad-a024-041c0d6d56a2/t2_tse_cor_p2_fs':'BC', 
    './studies/7ad5a1be-029e-4b7d-b57f-bef533e6ab05/t2_pdFS_384_cor':'SC',   
    './studies/7ad5a1be-029e-4b7d-b57f-bef533e6ab05/t1_se_cor_p2_384 fs':'BC', 
    './studies/cb5b937d-574a-4ba6-a9b2-7d656dc0471e/t2_pdFS_384_cor':'SC',   
    './studies/1742b962-1971-42b1-b0cc-d7bf2fc6c0b2/t2_pdFS_384_cor':'SC',   
    './studies/5e978e26-7cbd-49c8-8c42-3cc9f341ab1b/t2_pdFS_384_cor':'SC',   
    './studies/1717ad15-e2b8-4742-aca5-a20cf3e64149/t2_pdFS_384_cor':'SC',   
    './studies/903987c7-b56d-4a89-9797-357207696b3c/t2_tse_cor_320_p2_fs':'BC', 
    './studies/0965025e-e8db-426b-a9d8-acf7521f96a7/t2_pdFS_384_cor':'SC',   
    './studies/6b1ca623-c0da-46a5-a879-ca89da5120ba/t2_pdFS_384_cor':'SC',   
    './studies/6f951dec-675a-4f7a-81ec-1c46063f8d8d/t2_tse_cor_320_p2_fs':'BC', 
    './studies/0de49e22-e471-4432-bfb2-601906f3ddaa/t2_pdFS_384_cor':'SC',   
    './studies/fccf4d04-90c4-4620-bbd6-d56397f92e2e/t2_pdFS_384_cor':'SC',   
    './studies/0145ecbe-1c74-4824-a765-4583c2eed686/t2_pdFS_384_cor':'SC',   
    './studies/9412acbd-bd12-4c5b-97ac-e2c58afce419/t2_pdFS_384_cor':'SC',   
    './studies/7d107393-4514-4f4f-9860-0a7157302482/t2_pdFS_384_cor':'SC',   
    './studies/1b061437-100c-4f29-bf7d-1096ddc36d92/t2_pdFS_384_cor':'SC',   
    './studies/f2b9839f-6041-460e-a4d9-1edde64d210c/t2_pdFS_384_cor':'SC',   
    './studies/4fbd5e5c-30d0-4df5-961e-bda2cb2d82ec/t2_pdFS_384_cor':'SC',   
    './studies/01992d59-a873-4e3a-8f66-cde34467e4f4/t2_pdFS_384_cor':'SC',   
    './studies/01992d59-a873-4e3a-8f66-cde34467e4f4/t2_tse_cor_384_fs':'SC',   
    './studies/01992d59-a873-4e3a-8f66-cde34467e4f4/t1_se_cor_p2_384 fs':'BC',
    './studies/01992d59-a873-4e3a-8f66-cde34467e4f4/t1_vibe_fs_cor_256_3mm_dynamic':'SC',   
    './studies/de836dc0-e00b-4aae-846e-1624cb8510fa/t2_pdFS_384_cor':'SC',  
    './studies/0bf59c3d-fbcf-4c07-90d1-73c02195fe8b/t2_pdFS_384_cor':'SC',  
    './studies/fc03d23d-331a-47c4-a355-876eeeb12e42/t2_tse_cor_p2_fs':'BC', 
    './studies/58ba3d7f-9dd3-4653-9ebf-a2a7b9d27d19/t2_tse_cor_320_p2_fs':'BC', 
    './studies/de862aa9-2100-41f3-add5-bc661361b6bf/t2_tse_cor_p2_fs':'SC',  
    './studies/de862aa9-2100-41f3-add5-bc661361b6bf/t1_tse_cor_320 fs':'SC',  
    './studies/d4113333-0edd-4479-af7a-0b2fe91918a2/t1_tse_cor_320 fs':'BC', 
    './studies/54c2bc14-b7c1-4a90-b0fc-c6c63d36036a/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/74ed3149-db71-410a-ade7-1196413c921f/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/cf479b0d-9cca-4a49-8cac-035a20e8d580/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/ce4c98c6-8fdb-4554-99ec-bd057d48d3c5/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/d805e572-4b68-4843-bcf5-7b255ca5819d/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/174abafb-861b-428b-969b-0fcf7988f32d/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/0e686b5d-72fb-442a-a74d-d3c96ce32e39/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/457cb229-8837-4481-8970-8c7b13b749df/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/c3585eb9-7075-411b-9bad-3146ed0d9839/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/20a6f592-ecca-4916-9bee-f53a13a86dd3/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/4cd419f8-9e92-406d-aa41-ad6db4b110e6/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/54cfcdce-cb4a-45f9-95a7-22f1d9e92dd1/t2_pdFS_384_cor':'SC',  
    './studies/08b865cd-c204-480b-aa41-1620c21f601b/t2_tse_cor_320_p2_fs':  'BC', 
    './studies/8c5a06c0-48ff-4ddb-9799-23717ad7bbc1/t2_pdFS_384_cor':'SC',  
    './studies/c4ad35db-09af-4b13-bf9f-7d94c9e60f12/t2_pdFS_384_cor':'SC',  
    './studies/49912010-f2e2-4322-9962-5c5298f64ff8/t2_pdFS_384_cor':'SC',  
    './studies/6d16a6d4-a87f-4469-bd7b-51a4cf0e7f60/t2_tse_cor_320_p2_fs':'BC', 
    './studies/70ba8514-a440-44d4-8d70-fb161b3df6e8/t2_pdFS_384_cor':'SC',  
    './studies/255a46ad-75ed-4381-bbc2-3098fe0bb6ae/t2_pdFS_384_cor':'SC',  
    './studies/5c015cc5-9b9b-4886-b587-0174e70bddf9/t2_pdFS_384_cor':'SC',  
    './studies/f8c1dd83-9186-4023-b5d3-3c5e416e7ff1/t2_pdFS_384_cor':'SC',  
    './studies/19f25807-6dfd-4d95-9238-84d987e8cdf3/t2_pdFS_384_cor':'SC',  
    './studies/5bee1f6c-e355-43ef-9911-904d1d90bb6f/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/f082b6ef-353b-4732-89b0-346ac1864c06/t2_pdFS_384_cor':'SC',  
    './studies/894cec9a-1da9-44af-b229-eceb1eb7835e/t2_pdFS_384_cor':'SC',  
    './studies/6d51df56-9455-413a-b803-6086fb89e62f/t2_tse_cor_p2_fs': 'BC', 
    './studies/6d51df56-9455-413a-b803-6086fb89e62f/t1_tse_cor_320 fs': 'BC', 
    './studies/1acf45e4-ed6d-402e-816e-4f9a5f977a21/t2_pdFS_384_cor':'SC',  
    './studies/cf062365-6c1a-41eb-be22-122b599e9fe1/t2_pdFS_384_cor':'SC',  
    './studies/8795a410-f660-4c1e-b458-96ff7123e637/t2_pdFS_384_cor':'SC',  
    './studies/1a7a58d1-d8e8-44b9-9be0-02e002899529/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/f235e8b9-fa88-4dd6-97da-9388ed72b5d3/t2_pdFS_384_cor':'SC',  
    './studies/9b098eac-7388-4e71-9b92-5b09c445624a/t2_tse_cor_p2_fs': 'BC', 
    './studies/f235e8b9-fa88-4dd6-97da-9388ed72b5d3/t2_pdFS_384_cor':'SC',  
    './studies/9b098eac-7388-4e71-9b92-5b09c445624a/t2_tse_cor_p2_fs':  'BC', 
    './studies/9b098eac-7388-4e71-9b92-5b09c445624a/t1_tse_cor_320 fs': 'BC', 
    './studies/0704f8c2-5ebc-47d3-beae-0295cc6aa48a/t2_pdFS_384_cor':'SC',  
    './studies/78af98c7-f136-458a-9f76-3d65730203db/t2_pdFS_384_cor':'SC',  
    './studies/0ebf4489-d118-4c69-985e-32e9e434a167/t1_tse_cor_320 fs': 'BC', 
    './studies/0ebf4489-d118-4c69-985e-32e9e434a167/t2_tse_cor_p2_fs': 'BC', 
    './studies/0ebf4489-d118-4c69-985e-32e9e434a167/t1_tse_cor_320 fs': 'BC', 
     './studies/11ca0b08-861c-4b2a-a705-d9653d05e5be/t2_pdFS_384_cor':'SC',  
    './studies/6d17ab73-8f92-4914-bd48-78164ba3dedc/t2_pdFS_384_cor':'SC',  
    './studies/17580230-4fb5-409d-86b8-b8365d95ef49/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/b9a3c4d7-0a6f-4085-abbd-7edb0bc447c1/t2_pdFS_384_cor':'SC',  
    './studies/04901e2e-573e-4fba-b015-f4a86a2f36e6/t2_tse_cor_320_p2_fs': 'BC', 
    './studies/0c33649e-649c-43dd-a0f4-c00151621bcf/t2_pdFS_384_cor':'SC',  
    './studies/aae9272e-9e34-4ed1-98e5-7174f5368904/t2_pdFS_384_cor':'SC',  
'./studies/8efaa565-d21d-4fb0-a7f8-cd28f9de3f47/t2_tse_cor_320_p2_fs': 'BC', 
'./studies/4b1495b9-6a08-4b4e-97b5-86edc53d33ae/t2_pdFS_384_cor':'SC',  
'./studies/4b1495b9-6a08-4b4e-97b5-86edc53d33ae/t2_tse_cor_384 fs': 'BC', 
'./studies/28c1a5db-6d6b-4819-b672-4ab42c28fe30/t2_pdFS_384_cor':'SC',  
'./studies/b93bd916-7553-433b-bf5b-1ca1023fa00a/t2_pdFS_384_cor':'SC',  
'./studies/55108146-21ee-423f-a560-2a1467372c54/t2_pdFS_384_cor':'SC',  
'./studies/474a4482-82e9-48c2-8946-fe085e9cf3c9/t2_tse_cor_320_p2_fs': 'BC', 
'./studies/feaff1ff-238d-4a21-9856-984302a32a6c/t2_tse_cor_320_p2_fs': 'BC', 
'./studies/239199ca-8b6f-4c4a-9ef2-dd210ced1cd7/t2_pdFS_384_cor':'SC',  
'./studies/a7973f9e-e486-4219-b831-7931f2605590/t2_pdFS_384_cor':'SC',  
'./studies/3956346d-727d-4603-bc19-d1b33731d8a2/t2_tse_cor_320_p2_fs': 'BC', 
'./studies/ce3e2ac8-fbca-4947-897d-a0e98cd28ef8/t2_pdFS_384_cor': 'SC',  
'./studies/455fa6ba-7e1d-4d62-9917-f1d1f0448fb7/t2_pdFS_384_cor':'SC',  
'./studies/e312184d-c574-449f-9d37-0bf7b2880782/t2_pdFS_384_cor':'SC',  
'./studies/31825e89-4ece-4e6d-88f8-9766441b2f1e/t2_pdFS_384_cor':'SC',  
'./studies/1d90613b-832a-49c6-89d3-22286d63b423/t2_tse_cor_320_p2_fs': 'BC', 
'./studies/6f2f13d8-1437-4c77-a6ac-93b22502d2c9/t2_pdFS_384_cor':'SC',  
'./studies/4a6526b7-84ca-4b06-9549-6473001efa76/t2_pdFS_384_cor':'SC',  
'./studies/16688645-5ccb-42e6-83f1-a9dff5178ccb/t2_pdFS_384_cor':'SC',  

    
    
    
    
}

new_bs_dict = {
    
'./studies/d4113333-0edd-4479-af7a-0b2fe91918a2/t1_tse_cor_320 fs': -4,
'./studies/ce4c98c6-8fdb-4554-99ec-bd057d48d3c5/t2_tse_cor_320_p2_fs': 1,
'./studies/174abafb-861b-428b-969b-0fcf7988f32d/t2_tse_cor_320_p2_fs': -2, 
'./studies/457cb229-8837-4481-8970-8c7b13b749df/t2_tse_cor_320_p2_fs': -1, 
'./studies/4cd419f8-9e92-406d-aa41-ad6db4b110e6/t2_tse_cor_320_p2_fs': -1, 
'./studies/08b865cd-c204-480b-aa41-1620c21f601b/t2_tse_cor_320_p2_fs': -2, 
'./studies/6d16a6d4-a87f-4469-bd7b-51a4cf0e7f60/t2_tse_cor_320_p2_fs': -3, 
'./studies/5bee1f6c-e355-43ef-9911-904d1d90bb6f/t2_tse_cor_320_p2_fs': -2, 
'./studies/6d51df56-9455-413a-b803-6086fb89e62f/t2_tse_cor_p2_fs': 1,
'./studies/9b098eac-7388-4e71-9b92-5b09c445624a/t1_tse_cor_320 fs': 2, 
'./studies/0ebf4489-d118-4c69-985e-32e9e434a167/t1_tse_cor_320 fs': -5, 
'./studies/0ebf4489-d118-4c69-985e-32e9e434a167/t2_tse_cor_p2_fs': -5, 
'./studies/0ebf4489-d118-4c69-985e-32e9e434a167/t1_tse_cor_320 fs': -5, 

    
}