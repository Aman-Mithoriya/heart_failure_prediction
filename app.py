import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
# loading the trained model
pickle_in = open('classifier3.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Age,Sex,ChestPainType,RestingBp,Cholestrol,FastingBs,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope):   
    if (Sex== "M" or Sex=="m"):
        Sex=1
    else:
        Sex=0
    
    if (ChestPainType=="ATA" or ChestPainType=="Ata" or ChestPainType=="ata"):
        ChestPainType=1
    elif(ChestPainType=="Asy" or ChestPainType=="ASY" or ChestPainType=="asy"):
        ChestPainType=0
    elif(ChestPainType=="Nap" or ChestPainType=="NAP" or ChestPainType=="nap"):
        ChestPainType=2
    else:
        ChestPainType=3

    if(RestingECG=="Normal" or RestingECG=="normal"):
        RestingECG=1
    elif(RestingECG=="St" or RestingECG=="ST" or RestingECG=="st"):
        RestingECG=2
    else:
        RestingECG=0

    if(ExerciseAngina=="N" or ExerciseAngina=="n"):
        ExerciseAngina=0
    else:
        ExerciseAngina=1
    
    if(ST_Slope=="UP" or ST_Slope=="up"):
        ST_Slope=2
    elif(ST_Slope=="Down" or ST_Slope=="down"):
        ST_Slope=1
    else:
        ST_Slope=0

    Age=Age
    RestingBp=RestingBp
    Cholestrol=Cholestrol
    FastingBs=FastingBs
    MaxHR=MaxHR
    Oldpeak=Oldpeak
    prediction = classifier.predict([[Age,Sex,ChestPainType,RestingBp,Cholestrol,FastingBs,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])
    print(prediction)
    if prediction == [0]:
        pred = 'Failure'
    else:
        pred = 'Survived'
    return pred
    
def distribution_categorical_features(data, feature, target, colors) :
    sns.set(rc={'figure.figsize':(14,8.27)},font_scale=1.2)
    grouped_columns = sns.countplot(x=feature, hue=target, data=data,palette=colors)
    grouped_columns.set_title('Countplot for {} {}'.format(target, feature))
    # return grouped_columns 
    
def pred2(df):
    lbcode = LabelEncoder()
    arr=[]
    for i in a:
        k=0                                        
        for j in a[i]:
            if(pd.isna(j)):
                if(k not in arr):
                    arr.append(k)
            k+=1
        a=a.drop(arr)
        # df.info()
        s=0
        c=[]
        k=[]
        arr2=[]
        for j in a:
            x=a[j].dtype
            if(x=="object" ):
                a[j]=LabelEncoder().fit_transform(a[j])
                arr2.append(j)
                s+=1
            elif(x!="object"):
                k.append(j)
                c.append(s)
                s+=1
    




      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    # html_temp = """ 
    # <div style ="background-color:green;padding:10px"> 
    # <h1 style ="color:black;text-align:center;">Heart Diseases</h1> 
    # </div> 
    # """

    # st.markdown(html_temp, unsafe_allow_html = True) 
    st.header("HEART FAILURE PREDICTION :")

    app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction','Dataset'])
    if app_mode=='Home':
        st.title('Heart Failure Prediction :')  
        st.image('dataset-cover.jpg')
        st.subheader('ABSTRACT')
        st.write("Heart failure is a serious condition with high occurrence. With the advanced development in machine learning (ML), artificial intelligence (AI) and data science has been shown to be effective in assisting in decision making and predictions from the large quantity of data produced by the healthcare industry. One such reality is coronary heart disease. Various studies give the impression of predicting heart disease with ML techniques. There are many features/factors that lead to heart disease like age, blood pressure, sodium creatinine, ejection fraction etc. In this paper we propose a method to find important features by applying machine learning techniques. The work is to design and develop prediction of heart disease by feature ranking machine learning. Hence ML has a huge impact in saving lives and helping the doctors, widening the scope of research in actionable insights and driving complex decisions.")
        # st.markdown('Dataset :')
        # data=pd.read_csv('heart.csv')
        # st.write(data.head())
        # st.markdown('Applicant Income VS Loan Amount ')
        # st.bar_chart(data[['HeartDisease','Age']].head(20))

    elif app_mode=='Prediction':
        st.image('xx.jpg')

        st.subheader('Sir/Mam , You need to fill all necessary informations in order    to get a prediction of heart Diseases!')
        # st.sidebar.header("Informations about the patient:")
        result =""
        Age=st.slider('Age',20,80)
        Sex = st.selectbox('Sex',("M","F"))
        ChestPainType=st.selectbox('ChestPainType',("ATA","NAP","ASY","TA"))
        RestingBp=st.slider('RestingBp',0,200,0)
        Cholestrol=st.slider('Cholestrol',0,603,0)
        FastingBs=st.selectbox('FastingBs',("0","1"))
        RestingECG=st.selectbox('RestingBp',("Normal","LVH","ST"))
        MaxHR=st.slider('MaxHR',50,220)
        ExerciseAngina=st.selectbox( "ExerciseAngina",("Y","N"))
        Oldpeak=st.slider("Oldpeak",-3,7)
        ST_Slope=st.selectbox( "ST_Slope",("Up","Flat","Down"))
        if st.button("Predict"): 
            result = prediction(Age,Sex,ChestPainType,RestingBp,Cholestrol,FastingBs,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope) 
            if result == "Failure" :
                st.error(
                'Failure'
                )
                # st.markdown(
                # f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">',
                # unsafe_allow_html=True,)
            elif result == "Survived" :
                st.success(
                'Survived'
                )
            # st.markdown(
            #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            #     unsafe_allow_html=True,
            # )
            # st.success('Prediction {}'.format(result))
            # print(result)
    # dataset_mode= st.sidebar.selectbox('Select Page',['Home','Prediction','Dataset'])
    elif app_mode=="Dataset":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV",type=["csv"])
            
        if data_file is not None:

            file_details = {"filename":data_file.name, "filetype":data_file.type,
                                "filesize":data_file.size}
                
            st.write(file_details)
            df = pd.read_csv(data_file)
            df2=np.array(df)
            st.dataframe(df)
        



        if st.button("Prediction and visualize"):
            final=[]
            # result = prediction(df) 
            for i in df2 :
                result=prediction(Age=i[0],Sex=i[1],ChestPainType=i[2],RestingBp=i[3],Cholestrol=i[4],FastingBs=i[5],RestingECG=i[6],MaxHR=i[7],ExerciseAngina=i[8],Oldpeak=i[9],ST_Slope=i[10])
                final.append(result)
            final2=np.array(final)
            # st.dataframe(final)
            st.dataframe(final)

            # Show download button for the selected frame.
            # # Ref.: https://docs.streamlit.io/library/api-reference/widgets/st.download_button
            # csv = final.to_csv(index=False).encode('utf-8')
            # st.download_button(
            #     label="Download data as CSV",
            #     data=csv,
            #     file_name='selected_df.csv',
            #     mime='text/csv',
            # )
            HeartDisease=pd.DataFrame(final2,columns=["HeartDisease"])
            df1=pd.concat([HeartDisease,df])
            dict1 = {'X_axis': HeartDisease.iloc[:,0].values,
            'Y_axis': [i for i in range(len(final))] }
            # color="X_axis"
            df2 = pd.DataFrame(dict1)
            fig = px.bar(        
                df2,
                x = "X_axis",
                y = "Y_axis",
                title = "Bar Graph",
                color="X_axis",
                )
            st.plotly_chart(fig)
            # print(df1)
            # print(final2.reshape(-1,1))
            # fig=distribution_categorical_features(df1, "ST_Slope", "HeartDisease", ["#ff8fa3","#abc4ff"])
            # st.plot(fig)
            # fig2=distribution_categorical_features(df1, "ChestPainType", "HeartDisease", ["#c05299","#e7c8a0"])
            # st.plot(fig2)
            # fig3=distribution_categorical_features(df1, "RestingECG", "HeartDisease", ["#ED8975","#8FB9AA"])
            # st.plot(fig3)
            # plt.figure(figsize=(12, 7))
            # heartDisease_countplot = sns.countplot(x=df1.HeartDisease,palette=["#f8ad9d","#95d5b2"])
            # heartDisease_countplot.set_title("Distribution of Target 'Heart Disease'")
            # heartDisease_countplot.set_xticklabels(['No', 'Yes'], fontsize=20)

        

            

                



        #     # st.success('Prediction {}'.format(result))
        #     print(result)

            
        # when 'Predict' is clicked, make the prediction and store it 

     
if __name__=='__main__': 
    main()