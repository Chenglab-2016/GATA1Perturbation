  chrombpnet bias pipeline \
        -ibam ./sorted_H2.bam \
        -d "ATAC" \
        -g ./hg19.fa \
        -c ./hg19.chrom.sizes \
        -p ./peaks_no_blacklist.bed \
        -n ./output_negatives.bed \
        -fl ./splits/fold_0.json \
        -b 0.5 \
        -o ./bias_model/ \
        -fp Hudep2 \

chrombpnet pipeline \
        -ibam ./sorted_H2.bam \
        -d "ATAC" \
        -g ./hg19.fa \
        -c ./hg19.chrom.sizes \
        -p ./peaks_no_blacklist.bed \
        -n ./output_negatives.bed \
        -fl ./splits/fold_0.json \
        -b ./bias_model/models/Hudep2_bias.h5 \
        -o ./chrombpnet_model/

chrombpnet snp_score \
        -snps variant.tsv \
        -m ./chrombpnet_model/models/chrombpnet.h5 \
        -g ./hg19.fa \
        -op variant_

python chrombpnet_predict_onefilev.py \
  --model ./chrombpnet_model/models/chrombpnet.h5 \
  --csv inputs.csv \
  --inputlen 230 --outputlen 1 \
  --out-dir ./preds_csv