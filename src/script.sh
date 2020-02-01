um=user_m1
UM=USERM1

for fn in $(ls *aeif_cond_beta*.cu *aeif_cond_beta*.h); do
    fn1=$(echo $fn | sed "s/aeif_cond_beta/$um/")
    echo "$fn $fn1"
    cat $fn | sed "s/aeif_cond_beta/$um/g; s/AEIFCONDBETA/$UM/g" > \
		  $fn1
done
	      
um=user_m2
UM=USERM2

for fn in $(ls *aeif_cond_beta*.cu *aeif_cond_beta*.h); do
    fn1=$(echo $fn | sed "s/aeif_cond_beta/$um/")
    echo "$fn $fn1"
    cat $fn | sed "s/aeif_cond_beta/$um/g; s/AEIFCONDBETA/$UM/g" > \
		  $fn1
done

