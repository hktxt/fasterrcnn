import torch

def select_device(force_cpu=False):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        nCUDAs = torch.cuda.device_count()
        print('using {} GPU...'.format(nCUDAs))
        if nCUDAs:
            for i in range(nCUDAs):
                #print('{}/{}:{}'.format(i+1, nCUDAs, torch.cuda.get_device_name(i)))
                print('{}/{}:{}'.format(i+1, nCUDAs, torch.cuda.get_device_properties(i)))
    else:
        device = torch.device('cpu')
        print('using CPU...')
        
    return device