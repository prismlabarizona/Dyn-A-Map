import random
import numpy as np

def flip_weights_selected_subarray(weight_lines, num_rows, num_cols, row_blocks, col_blocks, flip_subarray=None, flip_rate=0.70):
    """
    Flips 5% of weights in the selected subarray only (by col, row order).
    flip_subarray: which subarray to flip (1-indexed), None to flip none.
    Returns the flat (col-major) list of weights (flipped as needed).
    """
    # Note the transpose: weights are indexed [col, row]
    weight_mat = np.array([float(x.strip()) for x in weight_lines]).reshape((num_cols, num_rows))  # <--- FIXED HERE!
    sub_idx = 1
    col_start = 0
    for cb, cblock in enumerate(col_blocks):
        row_start = 0
        for rb, rblock in enumerate(row_blocks):
            c0, c1 = col_start, col_start + cblock
            r0, r1 = row_start, row_start + rblock

            if flip_subarray == sub_idx:
                sub_indices = [
                    (c, r)
                    for c in range(c0, c1)
                    for r in range(r0, r1)
                    # no need to check != 0, all are 5000 or 15000
                ]
                num_to_flip = max(1, int(flip_rate * len(sub_indices)))
                if num_to_flip > 0:
                    flip_indices = random.sample(sub_indices, num_to_flip)
                    for (c, r) in flip_indices:
                        v = weight_mat[c, r]
                        weight_mat[c, r] = 5000.0 if v != 5000.0 else 15000.0
            row_start += rblock
            sub_idx += 1
        col_start += cblock
    # Write back in col-major order for netlist
    return [weight_mat[c, r] for c in range(num_cols) for r in range(num_rows)]


def mapLayer(layer1,layer2, LayerNUM,hpar,vpar,metal,T,H,L,W,D,eps,rho,weight_var,data_dir,spice_dir,run,flip_subarray=None): 

    # updating the resistivity for specific technology node
    l0 = 39e-9 # Mean free path of electrons in Cu
    d = metal # average grain size, equal to wire width
    p=0.25 # specular scattering fraction
    R=0.3 # probability for electron to reflect at the grain boundary
    alpha = l0*R/(d*(1-R)) # parameter for MS model
    dsur_scatt = 0.75*(1-p)*l0/metal # surface scattering
    dgrain_scatt = pow((1-3*alpha/2+3*pow(alpha,2)-3*pow(alpha,3)*np.log(1+1/alpha)),-1) # grain boundary scattering
    rho_new = rho * (dsur_scatt + dgrain_scatt) # new resistivity
    
    layer_w=open(spice_dir+'/'+f'layer{LayerNUM}run{run}.sp', "w") # open the layer subcircuit file for writing
    layer1_wb = layer1+1 # number of bitcell in a row including weights and bias
    
    layer_w.write(".SUBCKT layer"+ str(LayerNUM)+" vdd vss 0 ")

    for i in range(layer1):
        layer_w.write("in%d "%(i+1))
    
    for i in range(layer2):
        layer_w.write("out%d "%(i+1))

    
    if LayerNUM == 3:
        posw_r = open(data_dir+'/'+'posweight3_remapped.txt', "r")
        negw_r = open(data_dir+'/'+'negweight3_remapped.txt', "r")
    else:
        posw_r = open(data_dir+'/'+'posweight'+str(LayerNUM)+".txt", "r")
        negw_r = open(data_dir+'/'+'negweight'+str(LayerNUM)+".txt", "r")


    # ---------------------- Positive Weighted Array --------------------------
    layer_w.write("\n\n**********Positive Weighted Array**********\n")
    #posw_r=open(data_dir+'/'+'posweight'+str(LayerNUM)+".txt", "r")
    pos_lines = posw_r.readlines()
    posw_r.close()

    # Flip logic for Layer 1 only
    '''if LayerNUM == 3:
        num_rows, num_cols = 10, 84
        row_blocks = [10]
        col_blocks = [32, 32, 20]
        pos_weights = flip_weights_selected_subarray(
            pos_lines, num_rows, num_cols, row_blocks, col_blocks, flip_subarray=flip_subarray)
    else:
        pos_weights = [float(x.strip()) for x in pos_lines]'''

    pos_weights = [float(x.strip()) for x in pos_lines]

    n_hpar=1
    c=1
    r=1
    for idx, weight in enumerate(pos_weights):
        if weight != 0:
            if r < layer2+1:
                layer_w.write("Rwpos%d_%d in%d_%d sp%d_%d %f\n"% (c,r,c,r,c,r,weight))
                r+=1
            else:
                c+=1
                r=1
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    n_hpar+=1
                layer_w.write("Rwpos%d_%d in%d_%d sp%d_%d %f\n"% (c,r,c,r,c,r,weight))
                r+=1
        else:
            r+=1

    # ---------------------- Negative Weighted Array --------------------------
    layer_w.write("\n\n**********Negative Weighted Array**********\n\n")
    #negw_r=open(data_dir+'/'+'negweight'+str(LayerNUM)+".txt", "r")
    neg_lines = negw_r.readlines()
    negw_r.close()

    # Flip logic for Layer 1 only
    '''if LayerNUM == 3:
        num_rows, num_cols = 10, 84
        row_blocks = [10]
        col_blocks = [32, 32, 20]
        neg_weights = flip_weights_selected_subarray(
            neg_lines, num_rows, num_cols, row_blocks, col_blocks, flip_subarray=flip_subarray)
    else:
        neg_weights = [float(x.strip()) for x in neg_lines]'''
    
    neg_weights = [float(x.strip()) for x in neg_lines]
    n_hpar=1
    c=1
    r=1
    for idx, weight in enumerate(neg_weights):
        if weight != 0:
            if r < layer2+1:
                layer_w.write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (c,r,c,r,c,r,weight))
                r+=1
            else:
                c+=1
                r=1
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    n_hpar+=1
                layer_w.write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (c,r,c,r,c,r,weight))
                r+=1
        else:
            r+=1

    # ---------------------- Positive Biases --------------------------
    layer_w.write("\n\n**********Positive Biases**********\n\n")
    posb_r=open(data_dir+'/'+'posbias'+str(LayerNUM)+".txt", "r")
    r=1
    for line in posb_r:
        if (float(line)!=0):
            layer_w.write("Rbpos%d vd%d sp%d_%d %f\n"% (r,r,layer1_wb,r,float(line)))
            r+=1
        else:
            r+=1
    posb_r.close()

    # ---------------------- Negative Biases --------------------------
    layer_w.write("\n\n**********Negative Biases**********\n\n")
    negb_r=open(data_dir+'/'+'negbias'+str(LayerNUM)+".txt", "r")
    r=1
    for line in negb_r:
        if (float(line)!=0):
            layer_w.write("Rbneg%d vd%d sn%d_%d %f\n"% (r,r,layer1_wb,r,float(line)))
            r+=1
        else:
            r+=1
    negb_r.close()

    # ---------------------- Vertical Line Parasitic Resistances --------------------------
    layer_w.write("\n\n**********Parasitic Resistances for Vertical Lines**********\n\n")
    parasitic_res = rho_new*W/(metal*T)
    for i in range(layer1_wb):
        n_vpar=1
        c=i+1
        for j in range(layer2):
            r=j+1
            if (i == layer1): # only for the bias line
                if (j == 0):
                    layer_w.write("Rbias%d vdd vd%d %f\n"% (r,r,parasitic_res))
                elif (j == int(layer2*n_vpar/vpar+min((layer2%vpar)/n_vpar,1))):
                    layer_w.write("Rbias%d vdd vd%d %f\n"% (r,r,parasitic_res))
                    n_vpar+=1
                else:
                    layer_w.write("Rbias%d vd%d vd%d %f\n"% (r,j,r,parasitic_res))
            else: # the input connected vertical lines
                if (j == 0):
                    layer_w.write("Rin%d_%d in%d in%d_%d %f\n"% (c,r,c,c,r,parasitic_res))
                elif (j == int(layer2*n_vpar/vpar+min((layer2%vpar)/n_vpar,1))):
                    layer_w.write("Rin%d_%d in%d in%d_%d %f\n"% (c,r,c,c,r,parasitic_res))
                    n_vpar+=1
                else:
                    layer_w.write("Rin%d_%d in%d_%d in%d_%d %f\n"% (c,r,c,j,c,r,parasitic_res))
    
    # ---------------------- Horizontal Line Parasitic Resistances --------------------------
    layer_w.write("\n\n**********Parasitic Resistances for I+ and I- Lines****************\n\n")
    parasitic_res = rho_new*L/(metal*T)
    n_hpar=1
    for i in range(layer1_wb):
        c=i+1
        for j in range(layer2):
            r=j+1
            if (i == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)-1)):
                if (i == layer1):
                    layer_w.write("Rsp%d_%d sp%d_%d sp%d_p%d %f\n"% (c,r,c,r,r,n_hpar,parasitic_res))
                    layer_w.write("Rsn%d_%d sn%d_%d sn%d_p%d %f\n"% (c,r,c,r,r,n_hpar,parasitic_res))
                else:
                    layer_w.write("Rsp%d_%d sp%d_%d sp%d_p%d %f\n"% (c,r,c,r,r,n_hpar,parasitic_res))
                    layer_w.write("Rsn%d_%d sn%d_%d sn%d_p%d %f\n"% (c,r,c,r,r,n_hpar,parasitic_res))
                    if (j == layer2-1):
                        n_hpar+=1;
            else:
                layer_w.write("Rsp%d_%d sp%d_%d sp%d_%d %f\n"% (c,r,c,r,c+1,r,parasitic_res))
                layer_w.write("Rsn%d_%d sn%d_%d sn%d_%d %f\n"% (c,r,c,r,c+1,r,parasitic_res))

    # ---------------------- Differential Op-AMPS and Connecting Resistors --------------------------
    layer_w.write("\n\n**********Weight Differntial Op-AMPS and Connecting Resistors****************\n\n")
    for i in range(hpar):
        for j in range(layer2):
            layer_w.write("XDIFFw%d_p%d sp%d_p%d sn%d_p%d nin%d_%d diff%d\n"% (j+1,i+1,j+1,i+1,j+1,i+1,j+1,i+1,LayerNUM))
            layer_w.write("Rconn%d_p%d nin%d_%d nin%d 1m\n"% (j+1,i+1,j+1,i+1,j+1))

    # ---------------------- Neurons --------------------------
    layer_w.write("\n\n**********neurons****************\n\n")    
    for i in range(layer2):
        layer_w.write("Xsig%d nin%d out%d vdd 0 neuron\n"% (i+1,i+1,i+1))

    layer_w.write(".ENDS layer"+ str(LayerNUM))
    layer_w.close()

