datafile = 'sememe/hq_tokens.txt'  
sememe_vec = 'sememe/sememe-vec.txt'  

sememe_em = gensim.models.KeyedVectors.load_word2vec_format(sememe_vec, binary=False)  # 加载义原向量文件；

vec_zero = [0] * 200
vec_zero = np.array(vec_zero)
vec_zero = tf.convert_to_tensor(vec_zero)  
vec_zero = tf.cast(vec_zero, dtype=tf.float32)  
vec_zero = tf.expand_dims(vec_zero, 0)  
vec_zero_trans = tf.transpose(vec_zero)  
density_martix = tf.matmul(vec_zero_trans, vec_zero)  

with open(sememe_vec, encoding='UTF-8') as s:
sememe_lines = s.readlines()
for i in range(len(sememe_lines) - 1):
    sememe_line = sememe_lines[i+1].replace(' ', ',')   
    sememe_line = sememe_line.replace('\n', '')   
    sememe_line = sememe_line.split(',', 1)   
    sememe_char = sememe_line[0]  
    sememe_vector = sememe_em[sememe_char]    
    sememe_output = tf.convert_to_tensor(sememe_vector)   
    sememe_output = tf.expand_dims(sememe_output, 0)  
    sememe_output_trans = tf.transpose(sememe_output)  
    sememe_matrix = tf.matmul(sememe_output_trans, sememe_output)  
    density_martix = tf.add(density_martix, sememe_matrix)    

density_martix_expand = tf.expand_dims(density_martix, 1)  

sememe_feature = tf.layers.conv1d(density_martix_expand, 1, 1)  
sememe_feature = tf.squeeze(sememe_feature, axis=1)  
sememe_feature = tf.transpose(sememe_feature)  

input_shape = modeling.get_shape_list(input_ids, expected_rank=2)
batch_dim = input_shape[0]

sememe_feature = tf.tile(sememe_feature, [batch_dim, 1])  

output_layer = tf.concat([output_layer, sememe_feature], 1)  