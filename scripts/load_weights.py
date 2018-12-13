import numpy as np
import struct

def fuse_conv_batchnorm(layer_info,weights,biases,scales,r_mean,r_variance):
    n = int(layer_info['filters'])
    filter_size = (int(layer_info['size'])**2)*int(layer_info['c'])
    for i in range(n):
        den = (math.sqrt(r_variance[i])+0.000001)
        print(den)
        biases[i]= biases[i] - (scales[i] * r_mean[i])/(den)
        for j in range(filter_size):
            w_index = (i*filter_size) + j
            weights[w_index] = weights[w_index]*scales[i]/(den)
    return weights,biases


def extract_floats(fp,n_ele):
    n_bytes = n_ele*4
    res_bytes = fp.read(n_bytes)
    arr = struct.unpack_from('<'+str(n_ele)+'f',res_bytes)
    return list(arr)

def load_conv_weights(config,fp):
    print("Loading convolutional weights")
    n = int(config['filters'])
    c = int(config['c'])
    size = int(config['size'])
    num = n*c*size*size
    biases = extract_floats(fp,n)

    if 'batch_normalize' in config.keys():
        if config['batch_normalize']=='1':
            scales = extract_floats(fp,n)
            rolling_mean = extract_floats(fp,n)
            rolling_variance = extract_floats(fp,n)
            weights = extract_floats(fp,num)
        return (biases,scales,rolling_mean,rolling_variance,weights)
    else:
        weights = extract_floats(fp,num)
        return (biases,[1]*n,[0]*n,[0]*n,weights)

def parse_cfg_file(cfg_filename):
    cfg_file = open(cfg_filename,'r')
    cfg_lines = cfg_file.readlines()
    cfg_lines.remove('\n')

    layers = []
    curr_layer =  []
    for line in cfg_lines:
        if line[0]=='[' and line[-2]==']':
            layers.append(curr_layer)
            curr_layer = []
            #curr_layer.append(line[:-1])
        curr_layer.append(line[:-1])
    layer_dict = []

    for layer in layers[1:]:
        curr_layer_dict = {}
        #print(layer[1:-1])
        if layer[0][0]=='[' and layer[0][-1]==']' and layer[0][1:4]!='net':
        #if layer[0][1:-1]=='convolutional' or layer[0][1:-1]=='maxpool':
            print(layer[0][1:-1])
            curr_layer_dict['type'] = layer[0][1:-1]
            for line in layer[1:]:
                if not '=' in line:
                    continue
                key,value = line.split('=')
                if not key[0].isalpha():
                    value = float(value)

                curr_layer_dict[key] = value
            layer_dict.append(curr_layer_dict)
            if curr_layer_dict['type']=='convolutional':
                if len(layer_dict)>1:
                    n_prev = 1
                    while(n_prev<5):
                        if layer_dict[-1-n_prev]['type'] =='convolutional':
                            layer_dict[-1]['c'] = layer_dict[-1-n_prev]['filters']
                            break
                        else:
                            n_prev+=1
                else:
                    layer_dict[-1]['c']=3
    return layer_dict

def load_initial_info(fp):
    major = struct.unpack('<i',fp.read(4))
    minor = struct.unpack('<i',fp.read(4))
    revision = struct.unpack('<i',fp.read(4))
    iseen = struct.unpack('<Q',fp.read(8))
    return major,minor,revision,iseen

def load_network(weights_filename,layers_dict,n_layers):
    fp = open(weights_filename,'rb')
    init_info = load_initial_info(fp)

    weights = []
    for layer in layers_dict[:n_layers]:
        if layer['type'] =='convolutional':
            weights.append(load_conv_weights(layer,fp))
    return weights


