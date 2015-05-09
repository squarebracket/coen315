from VTC import VTC

vtc = VTC('vtc.csv', vin='V(in)', vout='V(out)')
print(vtc.vol, vtc.vil, vtc.vm, vtc.vih, vtc.voh, vtc.nml, vtc.nmh)
vtc.matplotlib('vtc.pdf')