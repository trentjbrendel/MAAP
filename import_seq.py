import numpy as np
import sys

def import_seq(seq_file, surf, mav):
    
    with open(seq_file, "r") as f:
        seq = f.read()
        seq = seq.split()

    k = [i for i,val in enumerate(seq) if val=='S']
    seq = seq[k[surf-1]:k[surf]]

    Typ_Sur = 1            
    CV_Coeff = np.zeros(11)            
    CV_Coeff[0]  = float(seq[1]) # Y Radius    

    for ele in seq:
        # if 'SPH' in ele:
        #     Typ_Sur = 1
        #     CV_Coeff = np.zeros(11)
        #     CV_Coeff[0]  = float(seq[1]) # Y Radius            
        if "ASP" in ele:
            Typ_Sur = 2
            CV_Coeff = np.zeros(11)
            CV_Coeff[0]  = float(seq[1]) # Y Radius
            if 'K' in seq:
                CV_Coeff[1]  = float(seq[seq.index('K')+1].replace(";",""))   # Conic Constant (K)
            elif 'K&' in seq:
                CV_Coeff[1]  = float(seq[seq.index('K&')+1].replace(";",""))   # Conic Constant (K)
            if 'A' in seq:
                CV_Coeff[2]  = float(seq[seq.index('A')+1].replace(";",""))   # 4th Order Coefficient (A);
            elif 'A&' in seq:
                CV_Coeff[2]  = float(seq[seq.index('A&')+1].replace(";",""))   # 4th Order Coefficient (A);
            if 'B' in seq:
                CV_Coeff[3]  = float(seq[seq.index('B')+1].replace(";",""))   # 6th Order Coefficient (B);
            elif 'B&' in seq:
                CV_Coeff[3]  = float(seq[seq.index('B&')+1].replace(";",""))   # 6th Order Coefficient (B);
            if 'C' in seq:
                CV_Coeff[4]  = float(seq[seq.index('C')+1].replace(";",""))   # 8th Order Coefficient (C);
            elif 'C&' in seq:
                CV_Coeff[4]  = float(seq[seq.index('C&')+1].replace(";",""))   # 8th Order Coefficient (C);
            if 'D' in seq:
                CV_Coeff[5]  = float(seq[seq.index('D')+1].replace(";",""))   # 10th Order Coefficient (D);
            elif 'D&' in seq:
                CV_Coeff[5]  = float(seq[seq.index('D&')+1].replace(";",""))   # 10th Order Coefficient (D);
            if 'E' in seq:
                CV_Coeff[6]  = float(seq[seq.index('E')+1].replace(";",""))   # 12th Order Coefficient (E);
            elif 'E&' in seq:
                CV_Coeff[6]  = float(seq[seq.index('E&')+1].replace(";",""))   # 12th Order Coefficient (E);
            if 'F' in seq:
                CV_Coeff[7]  = float(seq[seq.index('F')+1].replace(";",""))   # 14th Order Coefficient (F);
            elif 'F&' in seq:
                CV_Coeff[7]  = float(seq[seq.index('F&')+1].replace(";",""))   # 14th Order Coefficient (F);
            if 'G' in seq:
                CV_Coeff[8]  = float(seq[seq.index('G')+1].replace(";",""))   # 16th Order Coefficient (G);
            elif 'G&' in seq:
                CV_Coeff[8]  = float(seq[seq.index('G&')+1].replace(";",""))   # 16th Order Coefficient (G);
            if 'H' in seq:
                CV_Coeff[9]  = float(seq[seq.index('H')+1].replace(";",""))   # 18th Order Coefficient (H);
            elif 'H&' in seq:
                CV_Coeff[9]  = float(seq[seq.index('H&')+1].replace(";",""))   # 18th Order Coefficient (H);
            if 'J' in seq:
                CV_Coeff[10] = float(seq[seq.index('J')+1].replace(";",""))   # 20th Order Coefficient (J);
            elif 'J&' in seq:
                CV_Coeff[10] = float(seq[seq.index('J&')+1].replace(";",""))   # 20th Order Coefficient (J);
        elif "ZRN" in ele:
            Typ_Sur = 3
            CV_Coeff = np.zeros(69)
            CV_Coeff[0]  = float(seq[1]) # Y Radius
            CV_Coeff[1]  = float(seq[seq.index('NRADIUS')+1].replace(";","")) # Normalization Radius
            CV_Coeff[2]  = float(seq[seq.index('K')+1].replace(";","")) # Conic Constant (SCO K | C1)
            for i in range(2,67):
                s = 'C%d' % i
                if s in seq:
                    CV_Coeff[i+1] = float(seq[seq.index(s)+1].replace(";",""))
                else:
                    CV_Coeff[i+1] = 0
        else:
            pass            

        # Clear Aperture 크기 설정
        try:
            clearaperture = float(seq[seq.index('CIR')+1].replace(";",""))
        except:
            clearaperture = mav

        # 사각 어퍼쳐 x 크기
        try:
            rex = float(seq[seq.index('REX')+1].replace(";",""))
        except:
            rex = clearaperture
        # 사각 어퍼쳐 y 크기
        try:
            rey = float(seq[seq.index('REY')+1].replace(";",""))
        except:
            rey = clearaperture

        # Aperture Decenter(adx, ady) 확인
        try:
            adx = float(seq[seq.index('ADX')+1].replace(";",""))
        except:
            adx = 0
        try:
            ady = float(seq[seq.index('ADY')+1].replace(";",""))
        except:
            ady = 0
        ape_dec = [adx, ady]

        # Tilt (ade, bde, cde) 확인 degree
        try:
            ade = float(seq[seq.index('ADE')+1].replace(";",""))
        except:
            ade = 0
        try:
            bde = float(seq[seq.index('BDE')+1].replace(";",""))
        except:
            bde = 0
        try:
            cde = float(seq[seq.index('CDE')+1].replace(";",""))
        except:
            cde = 0

        tilt = [ade, bde, cde]

    if Typ_Sur == 0:
        print('Error : Surface type should be one of the followings: Sphere, Asphere, Zernike Polynomials.')
        sys.exit()
    else:
        pass

    return Typ_Sur, CV_Coeff, clearaperture, rex, rey, ape_dec, tilt