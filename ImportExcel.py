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
    dt = patientExcel[0][16][2]
    return getTime(dt, timestyle)
    
def getSedationStopTime(patient_number, timestyle = 'index'):
    patientExcel = getPatientExcel(patient_number)
    dt = patientExcel[0][17][2]
    return getTime(dt, timestyle)
    
def getWakeUpTime(patient_number, timestyle = 'index'):
    patientExcel = getPatientExcel(patient_number)
    dt = patientExcel[0][19][2]
    return getTime(dt, timestyle)