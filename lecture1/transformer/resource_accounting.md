1. Write down all the matrix multiplies in a Transformer forward pass.
    - transformer_blocks.ffn (swiglu)  
        + matrix multiplies: W2 @ (silu(W1@x) * (W3@x))  : 3 matrix multiplies
        + flops: for each sequence whose length is seq_len
            W1 @ x / W3@x: (2 * d_model)  * (seq_len * d_ff) 
            W2 @ intermedium : (2 * d_ff) * (seq_len * d_model)
    - transformer_blocks.attention 
        + q/k/v/o projection
            ++ matrix multiplies: q\k\v\o projection W_{q/k/v} @ x;  W_o @ o
            ++ flops: 4 * ( 2 * d_model ) * (seq_len * d_model)
        + rope
            ++ element-wise matrix multiplies: cos * x + sin_x * x
            ++ flops: 3 * d_model * seq_len
        + dot_product_attention
            ++ matrix multiplies: softmax(Q\top @ K / math.sqrt(d_model)) @ V
            ++ flops: (2 * d_model) * (seq_len * seq_len) + (2 * seq_len) * seq_len * d_model

    - transformer_blocks.rmsnorm
        + element-wise matrix multiplies: x * 1/rms_x * g 
        + flops: seq_len * seq_len * 2

    - lm_head
            + matrix multiplies: W_{h} @ logits
            flops: (2 * d_model) * (seq_len * vocab_size)


2.transformer accounting 
    - how many trainable parameters would gpt-2 XL with given config have?
        embedding: vocab_size*d_model = 50257 * 1600 = 80411200
        transformer_block: 
            single block:
                attention: W_q、W_k、W_v、W_o: 4 * d_model* d_model = 4* 1600 * 1600
                rms_norm*2: d_model * 2 = 1600 * 2
                swiglu: W1,W2,W3: d_model * d_ff*3 = 1600 * 6400 * 3 
                total = 40963200
            total blocks = 40963200 * 48 = 1966233600
        final_ln: d_model = 1600
        lm_head: d_model * vocab_size = 1600 * 50257 = 80411200

        total_lm : 80411200+1966233600 +1600 +80411200 = 2,127,057,600
        around 2B learnable parameter

        memory: 1 parameter using single-precision floating point costs 4 byte, so 2 Billion parameters cost 2*4=8 billion byte, aroud 8GB memory is required to just load such a model.

    - How many FLOPs do these matrix multiplies require intotal? Assume that our input sequence has context_length tokens.

        - transformer_blocks.ffn (swiglu)  
            + matrix multiplies: W2 @ (silu(W1@x) * (W3@x))  : 3 matrix multiplies
            + flops: for each sequence whose length is context_length
                W1 @ x / W3@x: (2 * d_model)  * (context_length * d_ff) * 2
                W2 @ intermedium : (2 * d_ff) * (context_length * d_model)
                total = 2 * 1600 * 1024 * 6400 * 2 + 2 * 6400 * 1024 * 1600 = 62914560000
        - transformer_blocks.attention 
            + q/k/v/o projection
                ++ matrix multiplies: q\k\v\o projection W_{q/k/v} @ x;  W_o @ o
                ++ flops: 4 * ( 2 * d_model ) * (context_length * d_model) 
                total = 4  * 2 * 1600 * 1024 * 1600 = 20971520000
            + rope
                ++ element-wise matrix multiplies: cos * x + sin_x * x
                ++ flops: 3 * d_model * context_length
                total = 3 * 1600 * 1024 = 4915200
            + dot_product_attention
                ++ matrix multiplies: softmax(Q\top @ K / math.sqrt(d_model)) @ V
                ++ flops: (2 * d_model) * (seq_len * seq_len) + (2 * seq_len) * seq_len * d_model
                total = 2 * 1600 * 1024 * 1024 + 2 * 1024 * 1024 * 1600 =6710886400

        - transformer_blocks.rmsnorm
            + element-wise matrix multiplies: x * 1/rms_x * g 
            + flops: context_length * d_model * 2
            total = 1024 * 1600 * 2 = 3276800
        
        total_transformer_blocks=(62,914,560,000+20,971,520,000+4915200+6,710,886,400+3276800) * 48 = 90605158400 * 48 = 4349047603200

        - final_ln
            + element-wise matrix multiplies:  x * 1/rms_x * g 
            + flops: context_length * d_model * 2
            total = 1024 * 1600 * 2 = 3276800

        - lm_head
                matrix multiplies W_{h} @ logits
                flops : (2 * d_model) * (context_length * vocab_size)
                total = 2 * 1600 * 1024 * 50257 = 164682137600
        
        total = 4,349,047,603,200 + 3,276,800 + 164,682,137,600 = 4,513,733,017,600

        FLOPs-to-parameters ratio : 4513733017600/2127057600 = 2122.05
        around 2*context_length

        example: 
            for MLP/FFN
                FLOPs_ffn = 2* d_model * seq_len * d_ff * 3 (SwiGLU)
                Params_ffn =  d_model * d_ff * 3 (SwiGLU)
            for dot product attention
                FLOPs_attn =(2 * d_model) * (seq_len * seq_len) + (2 * seq_len) * seq_len * d_model = 4 * seq_len *seq_len *d_model
                Params_atten =  4 * d_model* d_model 

        therefore 
            forward pass : 
                2 * N_param * L_seq
            backward pass:
                4 * N_param * L_seq
            training:
                6 * N_param * L_seq


    - Based on your analysis above,which parts of the model require the most FLOPs?
        SwiGLU

    - Repeat your analysis with GPT-2small (12layers,768 d_model,12heads), GPT-2medium(24 
    layers, 1024 d_model, 16 heads), and GPT-2 large(36layers, 1280 d_model,20 heads). As the 
    model size increases, which parts of the TransformerLM take up proportionally more or less of 
    the total FLOPs?

        using excel 
        GPT2 small
            parameter_name  count   single-module-to-total ratio    single-module-to-total ratio * num layer
            d_model	1600		
            context_length	1024		
            d_ff	6400		
            num_layers	48		
            vocab_size	50257		
            transformer_blocks.ffn (swiglu)  	62914560000	0.013938476	0.669046855
            W1 @ x / W3@x	41943040000		
            W2 @ intermedium	20971520000		
            transformer_blocks.attention	27687321600	0.006134018	0.294432886
            q/k/v/o projection	20971520000		
            rope	4915200		
            dot_product_attention	6710886400		
            transformer_blocks.rmsnorm	3276800	7.25962E-07	
            single_layer transformer block	90605158400	0.020073221	
            transformer_blocks	4.34905E+12	0.963514587	
            final_ln	3276800	7.25962E-07	
            lm_head	1.64682E+11	0.036484687	
            total	4.51373E+12

        GPT2 medium
            parameter_name  count   single-module-to-total ratio    single-module-to-total ratio * num layer
            d_model	1024		
            context_length	1024		
            d_ff	6400		
            num_layers	24		
            vocab_size	50257		
            transformer_blocks.ffn (swiglu)  	40265318400	0.029153899	0.69969358
            W1 @ x / W3@x	26843545600		
            W2 @ intermedium	13421772800		
            transformer_blocks.attention	12888047616	0.009331525	0.223956609
            q/k/v/o projection	8589934592		
            rope	3145728		
            dot_product_attention	4294967296		
            transformer_blocks.rmsnorm	2097152	1.51843E-06	
            single_layer transformer block	53155463168	0.038486943	
            transformer_blocks	1.27573E+12	0.923686632	
            final_ln	2097152	1.51843E-06	
            lm_head	1.05397E+11	0.07631185	
            total	1.38113E+12		

        GPT2 large
            parameter_name  count   single-module-to-total ratio    single-module-to-total ratio * num layer
            d_model	1280		
            context_length	1024		
            d_ff	6400		
            num_layers	24		
            vocab_size	50257		
            transformer_blocks.ffn (swiglu)  	50331648000	0.028105101	0.674522433
            W1 @ x / W3@x	33554432000		
            W2 @ intermedium	16777216000		
            transformer_blocks.attention	18794414080	0.010494767	0.251874405
            q/k/v/o projection	13421772800		
            rope	3932160		
            dot_product_attention	5368709120		
            transformer_blocks.rmsnorm	2621440	1.46381E-06	
            single_layer transformer block	69128683520	0.038601332	
            transformer_blocks	1.65909E+12	0.92643197	
            final_ln	2621440	1.46381E-06	
            lm_head	1.31746E+11	0.073566567	
            total	1.79084E+12		
	

        GPT2 xl
            parameter_name  count   single-module-to-total ratio    single-module-to-total ratio * num layer
            d_model	1600
            context_length	1024
            d_ff	6400
            num_layers	48
            vocab_size	50257
            transformer_blocks.ffn (swiglu)  	30198988800	0.056119347	0.673432166
            W1 @ x / W3@x	20132659200		
            W2 @ intermedium	10066329600		
            transformer_blocks.attention	8055422976	0.014969544	0.179634523
            q/k/v/o projection	4831838208		
            rope	2359296		
            dot_product_attention	3221225472		
            transformer_blocks.rmsnorm	1572864	2.92288E-06	
            single_layer transformer block	38255984640	0.071091814	
            transformer_blocks	4.59072E+11	0.853101763	
            final_ln	1572864	2.92288E-06	
            lm_head	79047426048	0.146895314	
            total	5.38121E+11		

        conclusion
            because context_length is constant, with the increase of d_model, SwiGLU's part-to-total ratio of FLOPs is increase, Attention's part-to-total ratio of FLOPs is decrease.
            other modules are also decrease just because d_model is decrease.


    - Take GPT-2XL and increase the context length to 16,384. How does the total FLOPs for one
     forwar dpass change? Ho wdo the relative contribution of FLOPs of the model component change?

        Total FOLPs increase to 1.49529E+14 from 4.51373E+12.
        The relative contributionof FLOPs of attention is increase to 65.92% from 29.44%.
        The relative contributionof FLOPs of ffn(SwiGLU) is decrease to 32.31% from 66.90%.


    
