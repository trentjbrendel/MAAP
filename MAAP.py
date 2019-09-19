# from MainAppClass_2 import MainApplication
# from MainAppClass import MainApplication
# from MainAppClass_new import MainApplication
from MainAppClass_new_out_in import MainApplication
import os
import time
import numpy as np
import csv

if __name__ == '__main__':
    # with open(os.getcwd() + '/info.txt', 'r') as f: # 단면
    #     seq = f.readline()[:-1]
    #     surf_i = int(f.readline()[:-1])
    #     surf_f = int(f.readline()[:-1])

    # surf_i = 2
    # surf_f = 13

    # savfile_final = 'MAAP_test_new.csv' # 최종 결과 저장할 파일 
    # fov = 0.5 # mm
    # sampling = 1001

    # Largan mav
    # mav_list_Largan = np.zeros(17)
    # mav_list_Largan[1] = 0.994328488432688 #SS
    # mav_list_Largan[2] = 1.00004226517969  #S2
    # mav_list_Largan[3] = 1.05520759221247  #S3
    # mav_list_Largan[4] = 1.08585422603525  #S4
    # mav_list_Largan[5] = 1.15311871409622  #S5
    # mav_list_Largan[6] = 1.17871867063153  #S6
    # mav_list_Largan[7] = 1.23928027298891  #S7
    # mav_list_Largan[8] = 1.27142407916431  #S8
    # mav_list_Largan[9] = 1.37767100137441  #S9
    # mav_list_Largan[10] = 1.51534610301609 #S10
    # mav_list_Largan[11] = 1.79392563270979 #S11
    # mav_list_Largan[12] = 2.11427012421069 #S12
    # mav_list_Largan[13] = 2.48052114776541 #S13
    # mav_list_Largan[14] = 2.8092009664794 #S14
    # mav_list_Largan[15] = 2.86631231064653 #S15
    # mav_list_Largan[16] = 3.0798555134919 #SI

    # PRI mav
    seq_PRI = 'PRI.seq'
    mav_list_PRI = np.zeros(10)
    mav_list_PRI[1] = 2.3004518712998 #SS
    mav_list_PRI[2] = 1.1695  #S2
    mav_list_PRI[3] = 1.0865  #S3
    mav_list_PRI[4] = 1.0525  #S4
    mav_list_PRI[5] = 0.9655  #S5
    mav_list_PRI[6] = 0.9995  #S6
    mav_list_PRI[7] = 1.239  #S7
    mav_list_PRI[8] = 2.5  #S8
    mav_list_PRI[9] = 2.5  #SI

    # N2A mav
    seq_N2A = 'N2A_13MP_Default2_SpaceDel_delsol.seq'
    mav_list_N2A = np.zeros(15)
    mav_list_N2A[1] =  0.81  #SS
    mav_list_N2A[2] =  0.80760218716995  #S2
    mav_list_N2A[3] =  0.7359819924528  #S3
    mav_list_N2A[4] =  0.718037532673862  #S4
    mav_list_N2A[5] =  0.659620495900313  #S5
    mav_list_N2A[6] =  0.694190714793386  #S6
    mav_list_N2A[7] =  0.897986294140941  #S7
    mav_list_N2A[8] =  1.40904128245974  #S8
    mav_list_N2A[9] =  1.62772958420772  #S9
    mav_list_N2A[10] = 1.99905754321341  #S10
    mav_list_N2A[11] = 2.24435943773655  #S11
    mav_list_N2A[12] = 2.25779851964986  #S12
    mav_list_N2A[13] = 2.3247767428667  #S13
    mav_list_N2A[14] = 2.63824238242409 #S14    
    
    t0_total = time.time()
    #################################
    # N2A
    t0 = time.time()
    app2 = MainApplication(seq=seq_N2A, fov=0.84, sampling=1001, ob_NA=0.3, savfile_final='MAAP_N2A', mav_list=mav_list_N2A)
    app2.run(surf_i=2, surf_f=11, TF=0.6, RF=0.4)
    app2.run_on_surf_for_TF_RF_list(surf = 11, TF_list=np.arange(0.5, 1.01, 0.1), RF_list=np.arange(0, 1.01, 0.2))
    app2.zernike_fit_for_TF_RF_list(surf = 11, TF_list=[0, 0.2, 0.4, 0.6, 0.8, 1], RF_list=[0, 0.2, 0.4, 0.6, 0.8, 1], num_term_1 = 100, num_term_2 = 66)
    
    try:
        app2.CV_stop()
    except:
        pass
    # 수행시간
    t = (time.time() - t0)
    print("--- Total time : %d min %d sec ---" %((t/60), (t%60)))

    # t0 = time.time()
    # # PRI
    # app = MainApplication(seq=seq_PRI, surf_i=2, surf_f=7, fov=0.84, sampling=1001, ob_NA=0.4, savfile_final='MAAP_PRI', mav_list=mav_list_PRI)
    # app.run()
    # try:
    #     app.CV_stop()
    # except:
    #     pass

    # # 수행시간
    # t = (time.time() - t0)
    # print("--- Total time : %d min %d sec ---" %((t/60), (t%60)))

    
    # 수행시간
    t = (time.time() - t0_total)
    print("--- Total time : %d min %d sec ---" %((t/60), (t%60)))



    


