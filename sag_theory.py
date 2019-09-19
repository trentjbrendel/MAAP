import numpy as np
from basic import cart2pol

# Asphere Surface
# Usage: sag = asphere(CV_Coeff, rho)
def asphere(CV_Coeff, rho):
    CV_Coeff = np.array(CV_Coeff)
    rho = np.array(rho)

    r = CV_Coeff[0]
    c = 1/r
    k = CV_Coeff[1] 
    A = CV_Coeff[2] 
    B = CV_Coeff[3] 
    C = CV_Coeff[4] 
    D = CV_Coeff[5] 
    E = CV_Coeff[6] 
    F = CV_Coeff[7] 
    G = CV_Coeff[8] 
    H = CV_Coeff[9] 
    I = CV_Coeff[10]
    
    sag = c*(rho**2)/(1+(1-(1+k)*c**2*rho**2)**0.5)+A*rho**4+B*rho**6+C*rho**8+D*rho**10+E*rho**12+F*rho**14+G*rho**16+H*rho**18+I*rho**20

    return sag

def asphere_xy(CV_Coeff, x, y):
    CV_Coeff = np.array(CV_Coeff)
    x = np.array(x)
    y = np.array(y)
    rho = np.sqrt(x**2 + y**2)

    r = CV_Coeff[0]
    c = 1/r
    k = CV_Coeff[1] 
    A = CV_Coeff[2] 
    B = CV_Coeff[3] 
    C = CV_Coeff[4] 
    D = CV_Coeff[5] 
    E = CV_Coeff[6] 
    F = CV_Coeff[7] 
    G = CV_Coeff[8] 
    H = CV_Coeff[9] 
    I = CV_Coeff[10]
    
    sag = c*(rho**2)/(1+(1-(1+k)*c**2*rho**2)**0.5)+A*rho**4+B*rho**6+C*rho**8+D*rho**10+E*rho**12+F*rho**14+G*rho**16+H*rho**18+I*rho**20

    return sag

def zernike(CV_Coeff, x, y):
    CV_Coeff = np.array(CV_Coeff)
    x = np.array(x)
    y = np.array(y)
    theta, rho = cart2pol(x,y)

    r = CV_Coeff[0] # Y Radius
    c = 1/r # Curvature
    Nor_Radius = CV_Coeff[1] # Normalization Radius
    k = CV_Coeff[2] # Conic Constant

    # Zernike Coefficient
    ZrnCoeff01 = CV_Coeff[3]  # 설계 시 사용

    ZrnCoeff02 = CV_Coeff[4]
    ZrnCoeff03 = CV_Coeff[5]  # 설계 시 사용

    ZrnCoeff04 = CV_Coeff[6]  # 설계 시 사용
    ZrnCoeff05 = CV_Coeff[7]  # 설계 시 사용
    ZrnCoeff06 = CV_Coeff[8]

    ZrnCoeff07 = CV_Coeff[9]
    ZrnCoeff08 = CV_Coeff[10]
    ZrnCoeff09 = CV_Coeff[11]  # 설계 시 사용
    ZrnCoeff10 = CV_Coeff[12]  # 설계 시 사용

    ZrnCoeff11 = CV_Coeff[13]  # 설계 시 사용
    ZrnCoeff12 = CV_Coeff[14]  # 설계 시 사용
    ZrnCoeff13 = CV_Coeff[15]  # 설계 시 사용
    ZrnCoeff14 = CV_Coeff[16]
    ZrnCoeff15 = CV_Coeff[17]

    ZrnCoeff16 = CV_Coeff[18]
    ZrnCoeff17 = CV_Coeff[19]
    ZrnCoeff18 = CV_Coeff[20]
    ZrnCoeff19 = CV_Coeff[21]  # 설계 시 사용
    ZrnCoeff20 = CV_Coeff[22]  # 설계 시 사용
    ZrnCoeff21 = CV_Coeff[23]  # 설계 시 사용

    ZrnCoeff22 = CV_Coeff[24]  # 설계 시 사용
    ZrnCoeff23 = CV_Coeff[25]  # 설계 시 사용
    ZrnCoeff24 = CV_Coeff[26]  # 설계 시 사용
    ZrnCoeff25 = CV_Coeff[27]  # 설계 시 사용
    ZrnCoeff26 = CV_Coeff[28]
    ZrnCoeff27 = CV_Coeff[29]
    ZrnCoeff28 = CV_Coeff[30]

    ZrnCoeff29 = CV_Coeff[31]
    ZrnCoeff30 = CV_Coeff[32]
    ZrnCoeff31 = CV_Coeff[33]
    ZrnCoeff32 = CV_Coeff[34]
    ZrnCoeff33 = CV_Coeff[35]  # 설계 시 사용
    ZrnCoeff34 = CV_Coeff[36]  # 설계 시 사용
    ZrnCoeff35 = CV_Coeff[37]  # 설계 시 사용
    ZrnCoeff36 = CV_Coeff[38]  # 설계 시 사용

    ZrnCoeff37 = CV_Coeff[39]  # 설계 시 사용
    ZrnCoeff38 = CV_Coeff[40]  # 설계 시 사용
    ZrnCoeff39 = CV_Coeff[41]  # 설계 시 사용
    ZrnCoeff40 = CV_Coeff[42]  # 설계 시 사용
    ZrnCoeff41 = CV_Coeff[43]  # 설계 시 사용
    ZrnCoeff42 = CV_Coeff[44]
    ZrnCoeff43 = CV_Coeff[45]
    ZrnCoeff44 = CV_Coeff[46]
    ZrnCoeff45 = CV_Coeff[47]

    ZrnCoeff46 = CV_Coeff[48]
    ZrnCoeff47 = CV_Coeff[49]
    ZrnCoeff48 = CV_Coeff[50]
    ZrnCoeff49 = CV_Coeff[51]
    ZrnCoeff50 = CV_Coeff[52]
    ZrnCoeff51 = CV_Coeff[53]  # 설계 시 사용
    ZrnCoeff52 = CV_Coeff[54]  # 설계 시 사용
    ZrnCoeff53 = CV_Coeff[55]  # 설계 시 사용
    ZrnCoeff54 = CV_Coeff[56]  # 설계 시 사용
    ZrnCoeff55 = CV_Coeff[57]  # 설계 시 사용

    ZrnCoeff56 = CV_Coeff[58]
    ZrnCoeff57 = CV_Coeff[59]
    ZrnCoeff58 = CV_Coeff[60]
    ZrnCoeff59 = CV_Coeff[61]
    ZrnCoeff60 = CV_Coeff[62]
    ZrnCoeff61 = CV_Coeff[63]  # 설계 시 사용
    ZrnCoeff62 = CV_Coeff[64]
    ZrnCoeff63 = CV_Coeff[65]
    ZrnCoeff64 = CV_Coeff[66]
    ZrnCoeff65 = CV_Coeff[67]
    ZrnCoeff66 = CV_Coeff[68]

    # Sphere Surface 함수
    Zrn_Z00=(c*(rho**2))/(1+np.sqrt(1-(1+k)*c**2*(rho**2)))

    # Zernike 식 계산을 위한 Normalized Radius 적용
    rho = rho/Nor_Radius

    # Zernike 식에 의한 각각의 Sag값
    Zrn_Z01=ZrnCoeff01*1

    Zrn_Z02=ZrnCoeff02*(rho*np.cos(theta))
    Zrn_Z03=ZrnCoeff03*(rho*np.sin(theta))

    Zrn_Z04=ZrnCoeff04*((rho**2)*np.cos(2*theta))
    Zrn_Z05=ZrnCoeff05*(2*(rho**2)-1)          
    Zrn_Z06=ZrnCoeff06*((rho**2)*np.sin(2*theta))

    Zrn_Z07=ZrnCoeff07*((rho**3)*np.cos(3*theta))
    Zrn_Z08=ZrnCoeff08*((3*(rho**3)-2*rho)*np.cos(theta))
    Zrn_Z09=ZrnCoeff09*((3*(rho**3)-2*rho)*np.sin(theta))
    Zrn_Z10=ZrnCoeff10*((rho**3)*np.sin(3*theta))

    Zrn_Z11=ZrnCoeff11*((rho**4)*np.cos(4*theta))
    Zrn_Z12=ZrnCoeff12*((4*(rho**4)-3*(rho**2))*np.cos(2*theta))
    Zrn_Z13=ZrnCoeff13*((6*(rho**4)-6*(rho**2)+1))
    Zrn_Z14=ZrnCoeff14*((4*(rho**4)-3*(rho**2))*np.sin(2*theta))
    Zrn_Z15=ZrnCoeff15*((rho**4)*np.sin(4*theta))

    Zrn_Z16=ZrnCoeff16*((rho**5)*np.cos(5*theta))
    Zrn_Z17=ZrnCoeff17*((5*(rho**5)-4*(rho**3))*np.cos(3*theta))
    Zrn_Z18=ZrnCoeff18*((10*(rho**5)-12*(rho**3)+3*rho)*np.cos(theta))
    Zrn_Z19=ZrnCoeff19*((10*(rho**5)-12*(rho**3)+3*rho)*np.sin(theta))
    Zrn_Z20=ZrnCoeff20*((5*(rho**5)-4*(rho**3))*np.sin(3*theta))
    Zrn_Z21=ZrnCoeff21*((rho**5)*np.sin(5*theta))

    Zrn_Z22=ZrnCoeff22*((rho**6)*np.cos(6*theta))
    Zrn_Z23=ZrnCoeff23*((6*(rho**6)-5*(rho**4))*np.cos(4*theta))
    Zrn_Z24=ZrnCoeff24*((15*(rho**6)-20*(rho**4)+6*(rho**2))*np.cos(2*theta))
    Zrn_Z25=ZrnCoeff25*((20*(rho**6)-30*(rho**4)+12*(rho**2)-1))
    Zrn_Z26=ZrnCoeff26*((15*(rho**6)-20*(rho**4)+6*(rho**2))*np.sin(2*theta))
    Zrn_Z27=ZrnCoeff27*((6*(rho**5)-5*(rho**4))*np.sin(4*theta))
    Zrn_Z28=ZrnCoeff28*((rho**6)*np.sin(6*theta))

    Zrn_Z29=ZrnCoeff29*((rho**7)*np.cos(7*theta))
    Zrn_Z30=ZrnCoeff30*((7*(rho**7)-6*(rho**5))*np.cos(5*theta))
    Zrn_Z31=ZrnCoeff31*((21*(rho**7)-30*(rho**5)+10*(rho**3))*np.cos(3*theta))
    Zrn_Z32=ZrnCoeff32*((35*(rho**7)-60*(rho**5)+30*(rho**3)-4*rho)*np.cos(theta))
    Zrn_Z33=ZrnCoeff33*((35*(rho**7)-60*(rho**5)+30*(rho**3)-4*rho)*np.sin(theta))
    Zrn_Z34=ZrnCoeff34*((21*(rho**7)-30*(rho**5)+10*(rho**3))*np.sin(3*theta))
    Zrn_Z35=ZrnCoeff35*((7*(rho**7)-6*(rho**5))*np.sin(5*theta))
    Zrn_Z36=ZrnCoeff36*((rho**7)*np.sin(7*theta))

    Zrn_Z37=ZrnCoeff37*((rho**8)*np.cos(8*theta))
    Zrn_Z38=ZrnCoeff38*((8*(rho**8)-7*(rho**6))*np.cos(6*theta))
    Zrn_Z39=ZrnCoeff39*((28*(rho**8)-42*(rho**6)+15*(rho**4))*np.cos(4*theta))
    Zrn_Z40=ZrnCoeff40*((56*(rho**8)-105*(rho**6)+60*(rho**4)-10*(rho**2))*np.cos(2*theta))
    Zrn_Z41=ZrnCoeff41*((70*(rho**8)-140*(rho**6)+90*(rho**4)-20*(rho**2)+1))
    Zrn_Z42=ZrnCoeff42*((56*(rho**8)-105*(rho**6)+60*(rho**4)-10*(rho**2))*np.sin(2*theta))
    Zrn_Z43=ZrnCoeff43*((28*(rho**8)-42*(rho**6)+15*(rho**4))*np.sin(4*theta))
    Zrn_Z44=ZrnCoeff44*((8*(rho**8)-7*(rho**6))*np.sin(6*theta))
    Zrn_Z45=ZrnCoeff45*((rho**8)*np.sin(8*theta))

    Zrn_Z46=ZrnCoeff46*((rho**9)*np.cos(9*theta))
    Zrn_Z47=ZrnCoeff47*((9*(rho**9)-8*(rho**7))*np.cos(7*theta))
    Zrn_Z48=ZrnCoeff48*((36*(rho**9)-56*(rho**7)+21*(rho**5))*np.cos(5*theta))
    Zrn_Z49=ZrnCoeff49*((84*(rho**9)-168*(rho**7)+105*(rho**5)-20*(rho**3))*np.cos(3*theta))
    Zrn_Z50=ZrnCoeff50*((126*(rho**9)-280*(rho**7)+210*(rho**5)-60*(rho**3)+5*rho)*np.cos(theta))
    Zrn_Z51=ZrnCoeff51*((126*(rho**9)-280*(rho**7)+210*(rho**5)-60*(rho**3)+5*rho)*np.sin(theta))      
    Zrn_Z52=ZrnCoeff52*((84*(rho**9)-168*(rho**7)+105*(rho**5)-20*(rho**3))*np.sin(3*theta))
    Zrn_Z53=ZrnCoeff53*((36*(rho**9)-56*(rho**7)+21*(rho**5))*np.sin(5*theta))
    Zrn_Z54=ZrnCoeff54*((9*(rho**9)-8*(rho**7))*np.sin(7*theta))
    Zrn_Z55=ZrnCoeff55*((rho**9)*np.sin(9*theta))

    Zrn_Z56=ZrnCoeff56*((rho**10)*np.cos(10*theta))
    Zrn_Z57=ZrnCoeff57*((10*(rho**10)-9*(rho**8))*np.cos(8*theta))
    Zrn_Z58=ZrnCoeff58*((45*(rho**10)-72*(rho**8)+28*(rho**6))*np.cos(6*theta))
    Zrn_Z59=ZrnCoeff59*((120*(rho**10)-252*(rho**8)+168*(rho**6)-35*(rho**4))*np.cos(4*theta))
    Zrn_Z60=ZrnCoeff60*((210*(rho**10)-504*(rho**8)+420*(rho**6)-140*(rho**4)+15*(rho**2))*np.cos(2*theta))
    Zrn_Z61=ZrnCoeff61*((252*(rho**10)-630*(rho**8)+560*(rho**6)-210*(rho**4)+30*(rho**2)-1))
    Zrn_Z62=ZrnCoeff62*((210*(rho**10)-504*(rho**8)+420*(rho**6)-140*(rho**4)+15*(rho**2))*np.sin(2*theta))
    Zrn_Z63=ZrnCoeff63*((120*(rho**10)-252*(rho**8)+168*(rho**6)-35*(rho**4))*np.sin(4*theta))
    Zrn_Z64=ZrnCoeff64*((45*(rho**10)-72*(rho**8)+28*(rho**6))*np.sin(6*theta))
    Zrn_Z65=ZrnCoeff65*((10*(rho**10)-9*(rho**8))*np.sin(8*theta))
    Zrn_Z66=ZrnCoeff66*((rho**10)*np.sin(10*theta))

    # 결과
    sag = \
        Zrn_Z00+\
        Zrn_Z01+\
        Zrn_Z02+Zrn_Z03+\
        Zrn_Z04+Zrn_Z05+Zrn_Z06+\
        Zrn_Z07+Zrn_Z08+Zrn_Z09+Zrn_Z10+\
        Zrn_Z11+Zrn_Z12+Zrn_Z13+Zrn_Z14+Zrn_Z15+\
        Zrn_Z16+Zrn_Z17+Zrn_Z18+Zrn_Z19+Zrn_Z20+Zrn_Z21+\
        Zrn_Z22+Zrn_Z23+Zrn_Z24+Zrn_Z25+Zrn_Z26+Zrn_Z27+Zrn_Z28+\
        Zrn_Z29+Zrn_Z30+Zrn_Z31+Zrn_Z32+Zrn_Z33+Zrn_Z34+Zrn_Z35+Zrn_Z36+\
        Zrn_Z37+Zrn_Z38+Zrn_Z39+Zrn_Z40+Zrn_Z41+Zrn_Z42+Zrn_Z43+Zrn_Z44+Zrn_Z45+\
        Zrn_Z46+Zrn_Z47+Zrn_Z48+Zrn_Z49+Zrn_Z50+Zrn_Z51+Zrn_Z52+Zrn_Z53+Zrn_Z54+Zrn_Z55+\
        Zrn_Z56+Zrn_Z57+Zrn_Z58+Zrn_Z59+Zrn_Z60+Zrn_Z61+Zrn_Z62+Zrn_Z63+Zrn_Z64+Zrn_Z65+Zrn_Z66
    return sag

def zernike_only(Zrn_Coeff, x, y):
    Zrn_Coeff = np.array(Zrn_Coeff)
    x = np.array(x)
    y = np.array(y)
    theta, rho = cart2pol(x,y)

    # Zernike Coefficient
    ZrnCoeff01 = Zrn_Coeff[0]  # Piston (constant)

    ZrnCoeff02 = Zrn_Coeff[1]  # Distortion - Tilt (x-axis)
    ZrnCoeff03 = Zrn_Coeff[2]  # Distortion - Tilt (y-axis)

    ZrnCoeff04 = Zrn_Coeff[3]  # Astigmatism, Primary (axis at 0° or 90°)
    ZrnCoeff05 = Zrn_Coeff[4]  # Defocus - Field curvature
    ZrnCoeff06 = Zrn_Coeff[5]  # Astigmatism, Primary (axis at ±45°)

    ZrnCoeff07 = Zrn_Coeff[6]  # Trefoil, Primary (x-axis)
    ZrnCoeff08 = Zrn_Coeff[7]  # Coma, Primary (x-axis)
    ZrnCoeff09 = Zrn_Coeff[8]  # Coma, Primary (y-axis)
    ZrnCoeff10 = Zrn_Coeff[9]  # Trefoil, Primary (y-axis)

    ZrnCoeff11 = Zrn_Coeff[10]  # Tetrafoil, Primary (x-axis)
    ZrnCoeff12 = Zrn_Coeff[11]  # Astigmatism, Secondary (axis at 0° or 90°)
    ZrnCoeff13 = Zrn_Coeff[12]  # Spherical Aberration, Primary
    ZrnCoeff14 = Zrn_Coeff[13]  # Astigmatism, Secondary (axis at ±45°)
    ZrnCoeff15 = Zrn_Coeff[14]  # Tetrafoil, Primary (y-axis)

    ZrnCoeff16 = Zrn_Coeff[15]  # Pentafoil, Primary (x-axis)
    ZrnCoeff17 = Zrn_Coeff[16]  # Trefoil, Secondary (x-axis)
    ZrnCoeff18 = Zrn_Coeff[17]  # Coma, Secondary (x-axis)
    ZrnCoeff19 = Zrn_Coeff[18]  # Coma, Secondary (y-axis)
    ZrnCoeff20 = Zrn_Coeff[19]  # Trefoil, Secondary (y-axis)
    ZrnCoeff21 = Zrn_Coeff[20]  # Pentafoil, Primary (y-axis)

    ZrnCoeff22 = Zrn_Coeff[21]  # 설계 시 사용
    ZrnCoeff23 = Zrn_Coeff[22]  # 설계 시 사용
    ZrnCoeff24 = Zrn_Coeff[23]  # 설계 시 사용
    ZrnCoeff25 = Zrn_Coeff[24]  # 설계 시 사용
    ZrnCoeff26 = Zrn_Coeff[25]
    ZrnCoeff27 = Zrn_Coeff[26]
    ZrnCoeff28 = Zrn_Coeff[27]

    ZrnCoeff29 = Zrn_Coeff[28]
    ZrnCoeff30 = Zrn_Coeff[29]
    ZrnCoeff31 = Zrn_Coeff[30]
    ZrnCoeff32 = Zrn_Coeff[31]
    ZrnCoeff33 = Zrn_Coeff[32]  # 설계 시 사용
    ZrnCoeff34 = Zrn_Coeff[33]  # 설계 시 사용
    ZrnCoeff35 = Zrn_Coeff[34]  # 설계 시 사용
    ZrnCoeff36 = Zrn_Coeff[35]  # 설계 시 사용

    ZrnCoeff37 = Zrn_Coeff[36]  # 설계 시 사용
    ZrnCoeff38 = Zrn_Coeff[37]  # 설계 시 사용
    ZrnCoeff39 = Zrn_Coeff[38]  # 설계 시 사용
    ZrnCoeff40 = Zrn_Coeff[39]  # 설계 시 사용
    ZrnCoeff41 = Zrn_Coeff[40]  # 설계 시 사용
    ZrnCoeff42 = Zrn_Coeff[41]
    ZrnCoeff43 = Zrn_Coeff[42]
    ZrnCoeff44 = Zrn_Coeff[43]
    ZrnCoeff45 = Zrn_Coeff[44]

    ZrnCoeff46 = Zrn_Coeff[45]
    ZrnCoeff47 = Zrn_Coeff[46]
    ZrnCoeff48 = Zrn_Coeff[47]
    ZrnCoeff49 = Zrn_Coeff[48]
    ZrnCoeff50 = Zrn_Coeff[49]
    ZrnCoeff51 = Zrn_Coeff[50]  # 설계 시 사용
    ZrnCoeff52 = Zrn_Coeff[51]  # 설계 시 사용
    ZrnCoeff53 = Zrn_Coeff[52]  # 설계 시 사용
    ZrnCoeff54 = Zrn_Coeff[53]  # 설계 시 사용
    ZrnCoeff55 = Zrn_Coeff[54]  # 설계 시 사용

    ZrnCoeff56 = Zrn_Coeff[55]
    ZrnCoeff57 = Zrn_Coeff[56]
    ZrnCoeff58 = Zrn_Coeff[57]
    ZrnCoeff59 = Zrn_Coeff[58]
    ZrnCoeff60 = Zrn_Coeff[59]
    ZrnCoeff61 = Zrn_Coeff[60]  # 설계 시 사용
    ZrnCoeff62 = Zrn_Coeff[61]
    ZrnCoeff63 = Zrn_Coeff[62]
    ZrnCoeff64 = Zrn_Coeff[63]
    ZrnCoeff65 = Zrn_Coeff[64]
    ZrnCoeff66 = Zrn_Coeff[65]

    # Zernike 식에 의한 각각의 Sag값
    Zrn_Z01=ZrnCoeff01*1

    Zrn_Z02=ZrnCoeff02*(rho*np.cos(theta))
    Zrn_Z03=ZrnCoeff03*(rho*np.sin(theta))

    Zrn_Z04=ZrnCoeff04*((rho**2)*np.cos(2*theta))
    Zrn_Z05=ZrnCoeff05*(2*(rho**2)-1)          
    Zrn_Z06=ZrnCoeff06*((rho**2)*np.sin(2*theta))

    Zrn_Z07=ZrnCoeff07*((rho**3)*np.cos(3*theta))
    Zrn_Z08=ZrnCoeff08*((3*(rho**3)-2*rho)*np.cos(theta))
    Zrn_Z09=ZrnCoeff09*((3*(rho**3)-2*rho)*np.sin(theta))
    Zrn_Z10=ZrnCoeff10*((rho**3)*np.sin(3*theta))

    Zrn_Z11=ZrnCoeff11*((rho**4)*np.cos(4*theta))
    Zrn_Z12=ZrnCoeff12*((4*(rho**4)-3*(rho**2))*np.cos(2*theta))
    Zrn_Z13=ZrnCoeff13*((6*(rho**4)-6*(rho**2)+1))
    Zrn_Z14=ZrnCoeff14*((4*(rho**4)-3*(rho**2))*np.sin(2*theta))
    Zrn_Z15=ZrnCoeff15*((rho**4)*np.sin(4*theta))

    Zrn_Z16=ZrnCoeff16*((rho**5)*np.cos(5*theta))
    Zrn_Z17=ZrnCoeff17*((5*(rho**5)-4*(rho**3))*np.cos(3*theta))
    Zrn_Z18=ZrnCoeff18*((10*(rho**5)-12*(rho**3)+3*rho)*np.cos(theta))
    Zrn_Z19=ZrnCoeff19*((10*(rho**5)-12*(rho**3)+3*rho)*np.sin(theta))
    Zrn_Z20=ZrnCoeff20*((5*(rho**5)-4*(rho**3))*np.sin(3*theta))
    Zrn_Z21=ZrnCoeff21*((rho**5)*np.sin(5*theta))

    Zrn_Z22=ZrnCoeff22*((rho**6)*np.cos(6*theta))
    Zrn_Z23=ZrnCoeff23*((6*(rho**6)-5*(rho**4))*np.cos(4*theta))
    Zrn_Z24=ZrnCoeff24*((15*(rho**6)-20*(rho**4)+6*(rho**2))*np.cos(2*theta))
    Zrn_Z25=ZrnCoeff25*((20*(rho**6)-30*(rho**4)+12*(rho**2)-1))
    Zrn_Z26=ZrnCoeff26*((15*(rho**6)-20*(rho**4)+6*(rho**2))*np.sin(2*theta))
    Zrn_Z27=ZrnCoeff27*((6*(rho**5)-5*(rho**4))*np.sin(4*theta))
    Zrn_Z28=ZrnCoeff28*((rho**6)*np.sin(6*theta))

    Zrn_Z29=ZrnCoeff29*((rho**7)*np.cos(7*theta))
    Zrn_Z30=ZrnCoeff30*((7*(rho**7)-6*(rho**5))*np.cos(5*theta))
    Zrn_Z31=ZrnCoeff31*((21*(rho**7)-30*(rho**5)+10*(rho**3))*np.cos(3*theta))
    Zrn_Z32=ZrnCoeff32*((35*(rho**7)-60*(rho**5)+30*(rho**3)-4*rho)*np.cos(theta))
    Zrn_Z33=ZrnCoeff33*((35*(rho**7)-60*(rho**5)+30*(rho**3)-4*rho)*np.sin(theta))
    Zrn_Z34=ZrnCoeff34*((21*(rho**7)-30*(rho**5)+10*(rho**3))*np.sin(3*theta))
    Zrn_Z35=ZrnCoeff35*((7*(rho**7)-6*(rho**5))*np.sin(5*theta))
    Zrn_Z36=ZrnCoeff36*((rho**7)*np.sin(7*theta))

    Zrn_Z37=ZrnCoeff37*((rho**8)*np.cos(8*theta))
    Zrn_Z38=ZrnCoeff38*((8*(rho**8)-7*(rho**6))*np.cos(6*theta))
    Zrn_Z39=ZrnCoeff39*((28*(rho**8)-42*(rho**6)+15*(rho**4))*np.cos(4*theta))
    Zrn_Z40=ZrnCoeff40*((56*(rho**8)-105*(rho**6)+60*(rho**4)-10*(rho**2))*np.cos(2*theta))
    Zrn_Z41=ZrnCoeff41*((70*(rho**8)-140*(rho**6)+90*(rho**4)-20*(rho**2)+1))
    Zrn_Z42=ZrnCoeff42*((56*(rho**8)-105*(rho**6)+60*(rho**4)-10*(rho**2))*np.sin(2*theta))
    Zrn_Z43=ZrnCoeff43*((28*(rho**8)-42*(rho**6)+15*(rho**4))*np.sin(4*theta))
    Zrn_Z44=ZrnCoeff44*((8*(rho**8)-7*(rho**6))*np.sin(6*theta))
    Zrn_Z45=ZrnCoeff45*((rho**8)*np.sin(8*theta))

    Zrn_Z46=ZrnCoeff46*((rho**9)*np.cos(9*theta))
    Zrn_Z47=ZrnCoeff47*((9*(rho**9)-8*(rho**7))*np.cos(7*theta))
    Zrn_Z48=ZrnCoeff48*((36*(rho**9)-56*(rho**7)+21*(rho**5))*np.cos(5*theta))
    Zrn_Z49=ZrnCoeff49*((84*(rho**9)-168*(rho**7)+105*(rho**5)-20*(rho**3))*np.cos(3*theta))
    Zrn_Z50=ZrnCoeff50*((126*(rho**9)-280*(rho**7)+210*(rho**5)-60*(rho**3)+5*rho)*np.cos(theta))
    Zrn_Z51=ZrnCoeff51*((126*(rho**9)-280*(rho**7)+210*(rho**5)-60*(rho**3)+5*rho)*np.sin(theta))      
    Zrn_Z52=ZrnCoeff52*((84*(rho**9)-168*(rho**7)+105*(rho**5)-20*(rho**3))*np.sin(3*theta))
    Zrn_Z53=ZrnCoeff53*((36*(rho**9)-56*(rho**7)+21*(rho**5))*np.sin(5*theta))
    Zrn_Z54=ZrnCoeff54*((9*(rho**9)-8*(rho**7))*np.sin(7*theta))
    Zrn_Z55=ZrnCoeff55*((rho**9)*np.sin(9*theta))

    Zrn_Z56=ZrnCoeff56*((rho**10)*np.cos(10*theta))
    Zrn_Z57=ZrnCoeff57*((10*(rho**10)-9*(rho**8))*np.cos(8*theta))
    Zrn_Z58=ZrnCoeff58*((45*(rho**10)-72*(rho**8)+28*(rho**6))*np.cos(6*theta))
    Zrn_Z59=ZrnCoeff59*((120*(rho**10)-252*(rho**8)+168*(rho**6)-35*(rho**4))*np.cos(4*theta))
    Zrn_Z60=ZrnCoeff60*((210*(rho**10)-504*(rho**8)+420*(rho**6)-140*(rho**4)+15*(rho**2))*np.cos(2*theta))
    Zrn_Z61=ZrnCoeff61*((252*(rho**10)-630*(rho**8)+560*(rho**6)-210*(rho**4)+30*(rho**2)-1))
    Zrn_Z62=ZrnCoeff62*((210*(rho**10)-504*(rho**8)+420*(rho**6)-140*(rho**4)+15*(rho**2))*np.sin(2*theta))
    Zrn_Z63=ZrnCoeff63*((120*(rho**10)-252*(rho**8)+168*(rho**6)-35*(rho**4))*np.sin(4*theta))
    Zrn_Z64=ZrnCoeff64*((45*(rho**10)-72*(rho**8)+28*(rho**6))*np.sin(6*theta))
    Zrn_Z65=ZrnCoeff65*((10*(rho**10)-9*(rho**8))*np.sin(8*theta))
    Zrn_Z66=ZrnCoeff66*((rho**10)*np.sin(10*theta))

    # 결과
    sag = \
        Zrn_Z01+\
        Zrn_Z02+Zrn_Z03+\
        Zrn_Z04+Zrn_Z05+Zrn_Z06+\
        Zrn_Z07+Zrn_Z08+Zrn_Z09+Zrn_Z10+\
        Zrn_Z11+Zrn_Z12+Zrn_Z13+Zrn_Z14+Zrn_Z15+\
        Zrn_Z16+Zrn_Z17+Zrn_Z18+Zrn_Z19+Zrn_Z20+Zrn_Z21+\
        Zrn_Z22+Zrn_Z23+Zrn_Z24+Zrn_Z25+Zrn_Z26+Zrn_Z27+Zrn_Z28+\
        Zrn_Z29+Zrn_Z30+Zrn_Z31+Zrn_Z32+Zrn_Z33+Zrn_Z34+Zrn_Z35+Zrn_Z36+\
        Zrn_Z37+Zrn_Z38+Zrn_Z39+Zrn_Z40+Zrn_Z41+Zrn_Z42+Zrn_Z43+Zrn_Z44+Zrn_Z45+\
        Zrn_Z46+Zrn_Z47+Zrn_Z48+Zrn_Z49+Zrn_Z50+Zrn_Z51+Zrn_Z52+Zrn_Z53+Zrn_Z54+Zrn_Z55+\
        Zrn_Z56+Zrn_Z57+Zrn_Z58+Zrn_Z59+Zrn_Z60+Zrn_Z61+Zrn_Z62+Zrn_Z63+Zrn_Z64+Zrn_Z65+Zrn_Z66
    return sag

def zernike_polynomials_xy(j, x, y):
    # j (order) is following OSA/ANSI standard indices
    # j = ( n ( n + 2 ) + m ) / 2
    j = abs(int(j))

    jin = j
    n = 0
    while j+1 > 0:
        j -= n
        n += 1
    n -= 2

    m = 2*jin - n*(n+2)

    n = int(n)
    m = int(m)
    
    x = np.array(x)
    y = np.array(y)
    theta, rho = cart2pol(x, y)

    if (n-m) % 2 == 1:
        val = 0
    else:
        if m >= 0:
            Theta_function = np.cos(m*theta)
        else:
            Theta_function = np.sin(-m*theta)

        m = abs(m)
        Rho_function = 0 
        for k in range(int((n-m)/2)+1):
            Rho_function += ((-1)**k * np.math.factorial(n-k))/(np.math.factorial(k)*np.math.factorial((n+m)/2-k)*np.math.factorial((n-m)/2-k)) * rho**(n-2*k)

        val = Rho_function * Theta_function

    return val

## Usage: sag = XYpol(CV_Coeff, x, y)
#def XYpol(CV_Coeff, x, y):
#    CV_Coeff = np.array(CV_Coeff)
#    x = np.array(x)
#    y = np.array(y)
#    theta, rho = cart2pol(x,y)
#    
#    r = CV_Coeff[0]
#    c = 1/r
#    k = CV_Coeff[1]     
#    
#    sag = c*(rho**2)/(1+(1-(1+k)*c**2*rho**2)**0.5)
#    
#    n = 10 # 차수 CV는 10차까지 사용
#    k = 2
#    for i in range(n):
#        for j in range(i+1):
#            sag += CV_Coeff[k] * x**(i-j) * y**j
#            k += 1        
#
#    return sag