.DEFAULT_GOAL := hh_sensitivity.svg

parameters_hh.npy:
	./analysis.py --sample $@

.PRECIOUS: hh_sensitivity_time.npy hh_sensitivity_membrane.npy hh_sensitivity_m.npy hh_sensitivity_n.npy hh_sensitivity_h.npy
hh_sensitivity_time.npy hh_sensitivity_membrane.npy: parameters_hh.npy
	./arbor_hh.py --parameter_file $<

hh_sensitivity.svg: hh_sensitivity_time.npy hh_sensitivity_membrane.npy
	./analysis.py --save $@

.PHONY: clean
clean:
	rm -f *svg *npy
