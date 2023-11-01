from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd

# List of the symptoms is listed here in list l1.

filename = 'Accuracy1.csv'
f = open(filename, 'w+')
f.writelines('\n Accuracy')
f.close()

l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

# List of Diseases is listed in list disease.

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
           'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
           'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
           'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
           'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
           'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
           'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
           'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
           'Osteoarthristis', 'Arthritis',
           '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
           'Urinary tract infection', 'Psoriasis', 'Impetigo']

l2 = []
for i in range(0, len(l1)):
    l2.append(0)

# Reading the training .csv file
df = pd.read_csv("training.csv")
DF = pd.read_csv('training.csv', index_col='prognosis')
df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                          'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

X = df[l1]
y = df[["prognosis"]]
np.ravel(y)

# Reading the  testing.csv file
tr = pd.read_csv("testing.csv")

# Using inbuilt function replace in pandas for replacing the values

tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                          'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)
tr.head()

X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)


def scatterplt(disea):
    x = ((DF.loc[disea]).sum())  # total sum of symptom reported for given disease
    x.drop(x[x == 0].index, inplace=True)  # dropping symptoms with values 0
    print(x.values)
    y = x.keys()  # storing name of symptoms in y
    print(len(x))
    print(len(y))
    plt.title(disea)
    plt.scatter(y, x.values)
    plt.show()


def scatterinp(sym1, sym2, sym3, sym4, sym5):
    x = [sym1, sym2, sym3, sym4, sym5]  # storing input symptoms in y
    y = [0, 0, 0, 0, 0]  # creating and giving values to the input symptoms
    if sym1 != 'Select Here':
        y[0] = 1
    if sym2 != 'Select Here':
        y[1] = 1
    if sym3 != 'Select Here':
        y[2] = 1
    if sym4 != 'Select Here':
        y[3] = 1
    if sym5 != 'Select Here':
        y[4] = 1
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.show()


root = Tk()
pred1 = StringVar()


def DecisionTree():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp = messagebox.askokcancel("System", "Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif (Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here"):
        pred1.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        clf3 = tree.DecisionTreeClassifier()
        clf3 = clf3.fit(X, y)
        y_pred = clf3.predict(X_test)
        print("Decision Tree")
        print("Accuracy")
        print(classification_report(y_test, y_pred))
        with open(filename, '+a') as f:
            f.writelines(f'\n mean_squared_error={mean_squared_error(y_test, y_pred, squared=False)}, '
                         f'accuracy_score={accuracy_score(y_test, y_pred)}')
            f.close()
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))
        print("Confusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)

        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

        for k in range(0, len(l1)):
            for z in psymptoms:
                if z == l1[k]:
                    l2[k] = 1

        inputtest = [l2]
        predict = clf3.predict(inputtest)
        predicted = predict[0]

        h = 'no'
        for a in range(0, len(disease)):
            if predicted == a:
                h = 'yes'
                break

        if h == 'yes':
            pred1.set(" ")
            pred1.set(disease[a])
        else:
            pred1.set(" ")
            pred1.set("Not Found")

        # printing scatter plot of input symptoms
        # printing scatter plot of disease predicted vs its symptoms
        scatterinp(Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get())
        scatterplt(pred1.get())


pred2 = StringVar()


def randomforest():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp = messagebox.askokcancel("System", "Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif (Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here"):
        pred1.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        clf4 = RandomForestClassifier(n_estimators=100)
        clf4 = clf4.fit(X, np.ravel(y))

        y_pred = clf4.predict(X_test)
        print("Random Forest")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))
        print("Confusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)

        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

        for k in range(0, len(l1)):
            for z in psymptoms:
                if z == l1[k]:
                    l2[k] = 1

        inputtest = [l2]
        predict = clf4.predict(inputtest)
        predicted = predict[0]

        h = 'no'
        for a in range(0, len(disease)):
            if predicted == a:
                h = 'yes'
                break
        if h == 'yes':
            pred2.set(" ")
            pred2.set(disease[a])
        else:
            pred2.set(" ")
            pred2.set("Not Found")
        # printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred2.get())


pred4 = StringVar()


def KNN():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp = messagebox.askokcancel("System", "Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif (Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here"):
        pred1.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        knn = knn.fit(X, np.ravel(y))
        y_pred = knn.predict(X_test)
        print("kNearest Neighbour")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))
        print("Confusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)

        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

        for k in range(0, len(l1)):
            for z in psymptoms:
                if z == l1[k]:
                    l2[k] = 1

        inputtest = [l2]
        predict = knn.predict(inputtest)
        predicted = predict[0]

        h = 'no'
        for a in range(0, len(disease)):
            if predicted == a:
                h = 'yes'
                break

        if h == 'yes':
            pred4.set(" ")
            pred4.set(disease[a])
        else:
            pred4.set(" ")
            pred4.set("Not Found")
        # printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred4.get())


pred3 = StringVar()


def NaiveBayes():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp = messagebox.askokcancel("System", "Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif (Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here"):
        pred1.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb = gnb.fit(X, np.ravel(y))

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        y_pred = gnb.predict(X_test)
        print("Naive Bayes")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))
        print("Confusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)

        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
        for k in range(0, len(l1)):
            for z in psymptoms:
                if z == l1[k]:
                    l2[k] = 1

        inputtest = [l2]
        predict = gnb.predict(inputtest)
        predicted = predict[0]

        h = 'no'
        for a in range(0, len(disease)):
            if predicted == a:
                h = 'yes'
                break
        if h == 'yes':
            pred3.set(" ")
            pred3.set(disease[a])
        else:
            pred3.set(" ")
            pred3.set("Not Found")
        # printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred3.get())


# Tk class is used to create a root window
root.configure(background='Ivory')
root.title('Smart Disease Predictor System')
root.resizable(0, 0)

Symptom1 = StringVar()
Symptom1.set("Select Here")

Symptom2 = StringVar()
Symptom2.set("Select Here")

Symptom3 = StringVar()
Symptom3.set("Select Here")

Symptom4 = StringVar()
Symptom4.set("Select Here")

Symptom5 = StringVar()
Symptom5.set("Select Here")
Name = StringVar()

prev_win = None


def Reset():
    global prev_win

    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    Symptom5.set("Select Here")
    NameEn.delete(first=0, last=100)
    pred1.set(" ")
    try:
        prev_win.destroy()
        prev_win = None
    except AttributeError:
        pass


def Exit():
    qExit = messagebox.askyesno("System", "Do you want to exit the system")

    if qExit:
        root.destroy()
        exit()


# Label for the name
NameLb = Label(root, text="Name of the Patient *", fg="Red", bg="Ivory")
NameLb.config(font=("Times", 15, "bold italic"))
NameLb.grid(row=6, column=0, pady=15, sticky=W)

# Creating Labels for the symptoms
S1Lb = Label(root, text="Symptom 1 *", fg="Black", bg="Ivory")
S1Lb.config(font=("Times", 15, "bold italic"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2 *", fg="Black", bg="Ivory")
S2Lb.config(font=("Times", 15, "bold italic"))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Black", bg="Ivory")
S3Lb.config(font=("Times", 15, "bold italic"))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="Black", bg="Ivory")
S4Lb.config(font=("Times", 15, "bold italic"))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="Black", bg="Ivory")
S5Lb.config(font=("Times", 15, "bold italic"))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

# Taking name as input from user
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

# Labels for Decision Tree algorithms
lrLb = Label(root, text="DecisionTree", fg="white", bg="red", width=20)
lrLb.config(font=("Times", 15, "bold italic"))
lrLb.grid(row=15, column=0, pady=10, sticky=W)

lrLb2 = Label(root, text="RandomForest", fg="white", bg="red", width=20)
lrLb2.config(font=("Times", 15, "bold italic"))
lrLb2.grid(row=16, column=0, pady=10, sticky=W)

lrLb3 = Label(root, text="KNN", fg="white", bg="red", width=20)
lrLb3.config(font=("Times", 15, "bold italic"))
lrLb3.grid(row=17, column=0, pady=10, sticky=W)

lrLb3 = Label(root, text="Naive Bayes", fg="white", bg="red", width=20)
lrLb3.config(font=("Times", 15, "bold italic"))
lrLb3.grid(row=18, column=0, pady=10, sticky=W)
OPTIONS = sorted(l1)

# Taking Symptoms as input from the dropdown from the user
S1 = OptionMenu(root, Symptom1, *OPTIONS)
S1.grid(row=7, column=1)

S2 = OptionMenu(root, Symptom2, *OPTIONS)
S2.grid(row=8, column=1)

S3 = OptionMenu(root, Symptom3, *OPTIONS)
S3.grid(row=9, column=1)

S4 = OptionMenu(root, Symptom4, *OPTIONS)
S4.grid(row=10, column=1)

S5 = OptionMenu(root, Symptom5, *OPTIONS)
S5.grid(row=11, column=1)

# Buttons for predicting the disease using Decision Tree algorithms
dst = Button(root, text="Check The Name of Disease Using DecisionTree", command=DecisionTree, bg="Red", fg="yellow")
dst.config(font=("Times", 15, "bold italic"))
dst.grid(row=6, column=3, padx=10)

rnf = Button(root, text="Check The Name of Disease Random Forest", command=randomforest, bg="Red", fg="yellow")
rnf.config(font=("Times", 15, "bold italic"))
rnf.grid(row=7, column=3, padx=10)

knn = Button(root, text="Check The Name of Disease Knn", command=KNN, bg="Red", fg="yellow")
knn.config(font=("Times", 15, "bold italic"))
knn.grid(row=8, column=3, padx=10)

nb = Button(root, text="Check The Name of Disease NaiveBayes", command=NaiveBayes, bg="Red", fg="yellow")
nb.config(font=("Times", 15, "bold italic"))
nb.grid(row=9, column=3, padx=10)

rs = Button(root, text="Reset Inputs", command=Reset, bg="yellow", fg="purple", width=15)
rs.config(font=("Times", 15, "bold italic"))
rs.grid(row=10, column=3, padx=10)

ex = Button(root, text="Exit System", command=Exit, bg="yellow", fg="purple", width=15)
ex.config(font=("Times", 15, "bold italic"))
ex.grid(row=11, column=3, padx=10)

# Showing the output of Decision Tree algorithm
t1 = Label(root, font=("Times", 15, "bold italic"), text="Decision Tree", height=1, bg="Light green"
           , width=40, fg="red", textvariable=pred1, relief="sunken").grid(row=15, column=1, padx=10)

t2 = Label(root, font=("Times", 15, "bold italic"), text="Random Forest", height=1, bg="Light green"
           , width=40, fg="red", textvariable=pred2, relief="sunken").grid(row=16, column=1, padx=10)

t3 = Label(root, font=("Times", 15, "bold italic"), text="kNearest Neighbour", height=1, bg="Light green"
           , width=40, fg="red", textvariable=pred3, relief="sunken").grid(row=17, column=1, padx=10)

t4 = Label(root, font=("Times", 15, "bold italic"), text="Naive Bayes", height=1, bg="Light green"
           , width=40, fg="red", textvariable=pred4, relief="sunken").grid(row=18, column=1, padx=10)

# calling this function because the application is ready to run
root.mainloop()
