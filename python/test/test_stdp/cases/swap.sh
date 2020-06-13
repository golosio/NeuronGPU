for i in $(seq 1 5); do
    fn=case${i}.py
    j=$(expr $i + 5)
    fn1=case${j}.py
    echo "$fn -> $fn1"
    cat $fn | sed 's/pre/POST/g;s/post/PRE/g' > tmp.py
    cat tmp.py | sed 's/POST/post/g;s/PRE/pre/g' > tmp1.py
    cat tmp1.py | sed 's/neuron_post, neuron_pre/neuron_pre, neuron_post/' > tmp2.py
    cat tmp2.py | sed 's/delay = 1.0/delay = 3.0/' > tmp3.py
    cat tmp3.py | sed 's/den_delay = 3.0/den_delay = 1.0/' > tmp4.py
    cat tmp4.py | sed 's/neuron_pre, {\"den_delay\"/neuron_post, {\"den_delay\"/' > $fn1
done
rm -f tmp.py
rm -f tmp1.py
rm -f tmp2.py
rm -f tmp3.py
