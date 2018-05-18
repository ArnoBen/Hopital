def getPatientExcel(patient_number):
    import pyexcel_ods as pe
    filepath = r"C:\Users\Arno\Documents\Patients\Fiches excel patients\Patient" + str(patient_number) + ".ods"
    patientExcel = pe.get_data(filepath)
    patientExcel = list(patientExcel.values())
    return patientExcel

def getTime(dateTime, timestyle='index'):
    try:
        hour = dateTime.hour
        minute = dateTime.minute
        second = dateTime.second       
        if timestyle == 'index'       : return 250*(second + 60*minute + 3600*hour)
        if timestyle == 'datetime'    : return [hour, minute, second]
        if timestyle == 'Time (hours)': return [hour + minute/60 + second/3600]
        if timestyle == 'Time (mins)' : return [hour*60 + minute + second/60]
        if timestyle == 'Time (secs)' : return [hour*3600 + minute*60 + second]
    except:
        return 0 #valeur arbitraire qui n'apparaît pas sur le graphe si la valeur lue n'est pas un datetime
def getPropofolTime(patient_number, timestyle = 'index'):
    patientExcel = getPatientExcel(patient_number)
    dt = patientExcel[0][10][2]
    return getTime(dt, timestyle)

def getSutureEndTime(patient_number, timestyle = 'index'):
    patientExcel = getPatientExcel(patient_number)
    try : dt = patientExcel[0][16][2]
    except : return 0
    return getTime(dt, timestyle)
    
def getSedationStopTime(patient_number, timestyle = 'index'):
    patientExcel = getPatientExcel(patient_number)
    try : dt = patientExcel[0][17][2]
    except : return 0
    return getTime(dt, timestyle)
    
def getWakeUpTime(patient_number, timestyle = 'index'):
    patientExcel = getPatientExcel(patient_number)
    try : dt = patientExcel[0][19][2]
    except : return 0
    return getTime(dt, timestyle)

def getPatientSummary(patient_number):
    import pyexcel_ods as pe
    filepath = r"C:\Users\Arno\Documents\Patients\Fiches excel patients\Patients utilisables.ods"    
    patientSummaryExcel = pe.get_data(filepath)
    patientSummaryExcel = list(patientSummaryExcel.values())
    return patientSummaryExcel
    
def getMaxRowLimit(patient_number): #Renvoie nombre de lignes avant que le signal soit médiocre.
    patientSummaryExcel = getPatientSummary(patient_number)
    MaxRowLimit = patientSummaryExcel[0][patient_number - 200][8]
    return MaxRowLimit
    
def getBadChannels(patient_number):
    patientSummaryExcel = getPatientSummary(patient_number)
    BadChannels = str(patientSummaryExcel[0][patient_number - 200][5])
    BadChannels = BadChannels.split(';')
    for i in range(len(BadChannels)) :
        BadChannels[i] = int(BadChannels[i]);
    return BadChannels

def getBadICAs(patient_number):
    patientSummaryExcel = getPatientSummary(patient_number)
    BadICAs = str(patientSummaryExcel[0][patient_number - 200][9])
    BadICAs = BadICAs.split(';')
    for i in range(len(BadICAs)) :
        BadICAs[i] = int(BadICAs[i]);
    return BadICAs