import os
import time
import random
import argparse
import pandas as pd
import torch
from tokenizers import Tokenizer
from model import ProGenForCausalLM
import pickle
os.environ['CUDA_CACHE_DISABLE'] = '1'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
########################################################################
# util
class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

########################################################################
# model

def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())
########################################################################
# sample

def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))


def truncate(sample, terminals):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample


def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)



########################################################################
# main


def main():
    
    models_2B = [ 'progen2-large' ]
    
    models =  models_2B 
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-large')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=38)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.9)
    parser.add_argument('--t', type=float, default=0.3)
    parser.add_argument('--max-length', type=int, default=50)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='2')
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'
    #args.device = 'cpu'
    device = torch.device(args.device)
    #ckpt = f'./checkpoints/{args.model}'
    ckpt = os.path.join(BASE_DIR, 'output', 'AMP_GPT_10', 'best_mode_eval')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(BASE_DIR, 'output', 'AMP_GPT_EMA_10', 'best_mode_eval')

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False
    args.fp16 = True

    # (3) load
    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file=os.path.join(BASE_DIR, 'checkpoints', 'tokenizer.json'))


    # (5) sample
    def generater_seq(device,model,tokenizer,context,num_samples,t,p,max_length):
        with torch.cuda.amp.autocast():
            completions = sample(device=device, model=model, tokenizer=tokenizer, context=context, pad_token_id=tokenizer.encode('<|pad|>').ids[0], num_return_sequences=num_samples, temp=t, top_p=p, max_length=max_length)
        torch.cuda.empty_cache()
        truncations = [truncate(completion, terminals=['1', '2']) for completion in completions]
        return truncations
    #(6) statistic the number of items in list
    def statistic_num(c_list):
        freq = {}
        for item in c_list:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
        return freq
    
    #loading beigining token
    with open(os.path.join(BASE_DIR, 'AMP_dipeptide_frequency_st.pkl'),'rb') as f:
        AA=pickle.load(f)
    
    #Generation        
    with print_time('sampling'):
        P= [0.95]
        t= [0.8]
        truncations = []
        for P_value in P:
            for t_value in t:
                args.p = P_value
                args.t = t_value
                AA_lst = random.choices(list(AA['Forward'].keys()), weights=AA['Forward'].values(), k=10000)
                AA_dict = statistic_num(AA_lst)
                print(AA_dict)
                for aa,num in AA_dict.items():
                    args.context = '1' +aa
                    args.num_samples = int(num)
                    if args.num_samples>1000:
                        args.num_samples = int(num)-1000
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                        truncations = truncations + truncations_
                        args.num_samples = 1000
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                        
                    else:
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                    truncations = truncations + truncations_
        results = [seq[1:-1] for seq in truncations]
        with open('AMP_generated_RL_statistic_p_t_1_aa.csv','w') as w:
            df = pd.DataFrame(results, columns=['Sequence'])
            df.to_csv(w,index=False)
    
    with print_time('sampling'):
        P= [0.95]
        t= [0.8]
        truncations = []
        for P_value in P:
            for t_value in t:
                args.p = P_value
                args.t = t_value
                AA_lst = random.choices(list(AA['Reverse'].keys()), weights=AA['Reverse'].values(), k=10000)
                AA_dict = statistic_num(AA_lst)
                for aa,num in AA_dict.items():
                    args.context = '2' +aa
                    args.num_samples = int(num)
                    if args.num_samples>1000:
                        args.num_samples = int(num)-1000
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                        truncations = truncations + truncations_
                        args.num_samples = 1000
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                        
                    else:
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                    truncations = truncations + truncations_
                    
        results = [seq[-2:0:-1] for seq in truncations]
        with open('AMP_generated_RL_statistic_p_t_2_aa.csv','w') as w:
            df = pd.DataFrame(results, columns=['Sequence'])
            df.to_csv(w,index=False)
    with print_time('sampling'):
        P= [0.95]
        t= [0.8]
        truncations = []
        truncations_r = []
        for P_value in P:
            for t_value in t:
                args.p = P_value
                args.t = t_value
                AA_lst = random.choices(list(AA['Forward'].keys()), weights=AA['Forward'].values(), k=5000)
                AA_dict = statistic_num(AA_lst)
                for aa,num in AA_dict.items():
                    args.context = '1' +aa
                    args.num_samples = int(num)
                    if args.num_samples>1000:
                        args.num_samples = int(num)-1000
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                        truncations = truncations + truncations_
                        args.num_samples = 1000
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                    else:
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                    truncations = truncations + truncations_
        
                AA_lst = random.choices(list(AA['Reverse'].keys()), weights=AA['Reverse'].values(), k=5000)
                AA_dict = statistic_num(AA_lst)
                for aa,num in AA_dict.items():
                    args.context = '2' +aa
                    args.num_samples = int(num)
                    if args.num_samples>1000:
                        args.num_samples = int(num)-1000
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                        truncations_r = truncations_r + truncations_
                        args.num_samples = 1000
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                        
                    else:
                        truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                    truncations_r = truncations_r + truncations_
                    
        results_r = [seq[-2:0:-1] for seq in truncations_r]
        results_f = [seq[1:-1] for seq in truncations]
        results = results_r + results_f
        with open('AMP_generated_RL_statistic_p_t_1_2_aa.csv','w') as w:
            df = pd.DataFrame(results, columns=['Sequence'])
            df.to_csv(w,index=False)


    '''
    with print_time('sampling'):
        P= [0.8,0.85,0.9,0.95]
        t= [0.05,0.1,0.15,0.2]
        truncations = []
        for P_value in P:
            for t_value in t:
                args.p = P_value
                args.t = t_value
                args.num_samples = 1
                AA_lst = random.choices(AA, k=625)
                for aa in AA_lst:
                    args.context = '1' +aa
                    truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                    truncations = truncations + truncations_
        results = [seq[1:-1] for seq in truncations]
        with open('test_results_generated_RL_p_t_1_aa_1.csv','w') as w:
            df = pd.DataFrame(results, columns=['Sequence'])
            df.to_csv(w,index=False)

    with print_time('sampling'):  
        P= [0.8,0.85,0.9,0.95]
        t= [0.05,0.1,0.15,0.2]  
        truncations = []
        for P_value in P:
            for t_value in t:
                args.p = P_value
                args.t = t_value
                args.num_samples = 1
                AA_lst = random.choices(AA, k=625)
                for aa in AA_lst:
                    args.context = '2' +aa
                    truncations_ = generater_seq(device,model,tokenizer,args.context,args.num_samples,args.t,args.p,args.max_length)
                    truncations = truncations + truncations_
        results = [seq[-2:0:-1] for seq in truncations]

        with open('test_results_generated_RL_p_t_2_aa_1.csv','w') as w:
            df = pd.DataFrame(results, columns=['Sequence'])
            df.to_csv(w,index=False)
    '''



if __name__ == '__main__':
    main()
    print('done.')
