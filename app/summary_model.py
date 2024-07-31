from langchain_community.llms import LlamaCpp
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
StreamingStdOutCallbackHandler,
)
# Define the path to your Llama.cpp model (assuming it's still functional)
def create_llm():
   callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
   n_gpu_layers = 35  # Change this value based on your model and your GPU VRAM pool.
   n_ctx = 4096
   n_batch = 4096
   n_threads=4  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
   llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q3_K_L.gguf",
    n_gpu_layers=n_gpu_layers,
    n_ctx=n_ctx,
    n_batch=n_batch,
    n_threads=n_threads,
    callback_manager=callback_manager,
    verbose=True,
    temperature=0.1,
    )
   return llm
def summarize(text):
   llm=create_llm()
   text_splitter = CharacterTextSplitter()
   texts = text_splitter.split_text(text)
   docs = [Document(page_content=t) for t in texts] 
   # Text summarization
   chain = load_summarize_chain(llm, chain_type='map_reduce')
   summary=chain.invoke(docs)
   return summary.get('output_text')

# Example usage (use with caution)
if __name__ == "__main__":
   text = """CONSUMER DISPUTES REDRESSAL FORUM -II UDYOG SADAN C C 22 23
QUTUB INSTITUTIONNAL AREA BEHIND QUTUB HOTEL NEW DELHI 110016
Complaint Case No. CC/103/2016
( Date of Filing : 06 Apr 2016 )
1. ADITI GUPTA
B-33, PANCHSHEEL ENCLAVE, NEW DELHI 110017
...........Complainant(s)
Versus
1. MOTOROLA MOBILITY INDIA PVT LTD
7A 12th FLOOR, TOWER-D DLF CYBER GREENS PHASE -III GURGAON
122002. HARYANA
............Opp.Party(s)
BEFORE:
HON'BLE MS. REKHA RANI PRESIDENT
KIRAN KAUSHAL MEMBER
For the Complainant:
None
For the Opp. Party:
None
Dated : 05 Dec 2019
Final Order / Judgement
                                                      DISTRICT CONSUMER DISPUTES REDRESSAL FORUM-II
Udyog Sadan, C-22 & 23, Qutub Institutional Area
(Behind Qutub Hotel), New Delhi-110016
Case No. 103/2016
Ms. Aditi Gupta
D/o Mr. Atul  Kumar
Cause Title/Judgement-Entry
https://cms.nic.in/ncdrcusersWeb/search.do?method=loadSearchPub
1 of 6
08-05-2024, 16:32
B-33,  Panchsheel Enclave,
New Delhi-110017                                                           ….Complainant
Versus
1.       M/s Motorola Mobility India Pvt.
           Building No.7A, 12th Floor,
          Tower D, DLF Cyber Greens Phase-III,
          Gurgaon 122002, Haryana.
2.       Services and Communication Solutions,
          ( Motorola Authorized Service Centre),
          A-180, Sukhdev Vihar Market,
          Kotla Mubarakpur, Bhishma Pitamah Marg,
          New Delhi-110003
3.       Flipkart International Private Limited,
          Vaishnavi Summit, Ground Floor,
          7th Main, 80 Feet Road, 3rd Block,
Cause Title/Judgement-Entry
https://cms.nic.in/ncdrcusersWeb/search.do?method=loadSearchPub
2 of 6
08-05-2024, 16:32
          Koramangla Industrial Layout,
          Bangalore-560034, Karnataka                                 ….Opposite Party
                                                Date of Institution            :         06.04.16   Date of Order                  :         05.12.19     
Coram:
Ms. Rekha Rani, President
Ms. Kiran Kaushal, Member
Ms. Kiran Kaushal, Member
ORDER
1. Brief facts of the complaint as stated by the complainant are:
1. The brother of the complainant purchased a mobile phone, Motorola X (2nd Gen) from Flipkart International Private
Limited (OP No.3) which is an online shopping website. The product was delivered on 26.12.2014. Thereafter the
complainant paid to her brother the entire amount charged, to his credit card and starting using the phone.   Copy of
invoice dated 22.12.2014 is annexed as Annexure-A.
1. It is averred that the complainant’s phone stopped working on or around 30.11.2015 and as per  the suggestion of the
customer care executive the mobile phone was given to Services and Communication Solutions (OP No.2) which is an
authorized service centre of the manufacturing company i.e.  M/s Motorola Mobility India Pvt. (OP No.1). It was
informed to the complainant that delivery of the repaired phone will be given to the complainant on 03.12.2015. The
complainant visited the service centre on 03.12.2015 and was told that the phone has been sent to replace the
Cause Title/Judgement-Entry
https://cms.nic.in/ncdrcusersWeb/search.do?method=loadSearchPub
3 of 6
08-05-2024, 16:32
motherboard which would further take 15 days. It is next averred on 11.12.15 the complainant received a call from OP
No.2 stating that they have repaired the phone and the complainant can collect the same.
1. On 12.12.15 the complainant went to the OP No.2 to take the phone but was appalled at the condition of the phone as it
had lots of scratches and dents on the back panel of the phone. The photographs of the phone taken at the premises of
OP No.2 are annexed as Annexure-C. The complainant refused to take the phone in the said condition and asked OP
No.2 to replace the phone. OP No.2 refused to replace the phone. Complainant  contacted the customer care of OP
No.1 and wrote various emails to OP No.1 & OP No.2 to resolve the issue but all in vain. The complainant thus
approached the Forum with the prayer to direct the OPs to replace the complainant’s phone or return the amount paid
to OPs. Further it is prayed that OPs be directed to pay sum of Rs.20,000/- on account of damages for mental tension
and Rs.16,000/- towards litigation expenses   
2. OP No.3 resisted the complaint stating inter-alia that OP No.3 is engaged in the business of online market and is only
providing platform to facilitate transaction between buyers and sellers. OP No.3 further submits that the sole grievance of
the complainant in the present complaint relates to the after sale service which was not provided to her when her phone had
some defects after she used the mobile for about one year. It is next submitted that the averments made in the complaint
neither point out any specific grievance against OP No.3 nor any specific relief had been claimed against OP No.3 therefore
on this ground alone the complaint is liable to be dismissed.
3. OP No.1 and OP No.2 were served notices. As none appeared on their behalf to contest they were proceeded against exparte
on 08.08.2016.
4. Averments made by the complainant qua OP No.1 and OP No.2 and evidence led by the complainant have remained
uncontroverted and unchallenged. Hence, there is no reason to disbelieve the version of the complainant.
5. The complainant to support her case has annexed copy of invoice dated 22.12.2014, Standard Chartered Bank Credit Card
Statement dated 12.01.2015 and RBL Bank Statement of January 2015 wherein it is shown that the complainant paid the
money  to her brother for the purchase of the mobile  phone in question which are annexed as Annexure-A.
6. Complainant has filed rejoinder and evidence by way of affidavit. Evidence by way of affidavit of Sh. Satyajeet
Bhattacharya, AR is filed on behalf of OP-3.
7. Written arguments are filed on behalf of the complainant and OP No.3.
8. We have heard the arguments on behalf of the complainant and OP No.3 and perused the materials placed on record.
9. The complainant is the beneficiary/user of the goods/mobile phone.  Further, it is not disputed that services of OP-2 were
solicited by the complainant for repairs. Hence being a consumer the complainant filed the complaint against OPs stating
after using her phone for about eleven months, it stopped working and did not switch ‘On’. The complainant gave her phone
at the service centre of OP No.2 as there were certain on and off issues with her phone. She could not turn the phone ‘On’.
The service note is annexed as Annexure-B wherein the problems reported by the customer are mentioned. For ready
reference they are reproduced below:-  
Cause Title/Judgement-Entry
https://cms.nic.in/ncdrcusersWeb/search.do?method=loadSearchPub
4 of 6
08-05-2024, 16:32
Problem Reported by the Customer:
Problem                       Problem                       Problem description
Category                     Subcategory                Any problem where the unit appears
                                                                         to turn on or off unexpectedly tests
Power              Power ON- Can’t                    or does not turn on or off properly    
On/off                         turn phone on
Issues
Job remarks     h/s dead, (freeze on moto), h/s dead, h/s rcvd w/o sim, mmc,
scratches on screen guard, internal condition not check, (unable to check)
10. On perusal of the Service Note dated 03.12.15 it is noticed that the major problem with the phone was that of the on and off
issue and there were some scratches on the screen guard. No scratches or dents were noticed or noted in Service Note of OP
No.2. Therefore we hold that there were no scratches or dents on the back panel when the phone was submitted with OP-2
for repairs.
11. The complainant is silent regarding the changing of the mother board, regarding the working condition of the phone,
regarding any payment made by her to the service centre or any payment demanded by the service centre. Therefore we have
no reasons to record a finding that the phone was not functional. The complainant’s right is only limited to getting the
aesthetics of the phone restored to the condition in which it was delivered to OP No.2. Therefore, grievance of the
complainant that when she went to collect the phone from the service centre there were lots of scratches and dents on the
back panel of the phone is genuine and we uphold the same.
Cause Title/Judgement-Entry
https://cms.nic.in/ncdrcusersWeb/search.do?method=loadSearchPub
5 of 6
08-05-2024, 16:32
12. Further it is a settled law that an equipment or machinery cannot be ordered to be replaced if it can be repaired. The same
was held in Shiv Prasad Paper Industry Vs. Senior Machinery Company  [ I (2006) CPJ 92 Hon’ble National Commission].
13. As regard OP No.3, OP No.3 has no liability in the  present case as it is a Online Shopping Website and has no role in the
after sale service.
14. In view of the settled law and facts of the case the complaint is partly allowed and OP No.1 and OP No.2 are directed to
repair the phone of the complainant and return it back to her in good condition. Additionally OP No.1 and OP No.2 are
directed to pay Rs.7500/- to the complainant towards compensation and litigation cost. The OP No.1 and OP No.2 shall
comply with this order within 45 days from the date  of receipt of copy of this order. Failing which OP-1 and OP No.2 shall
pay the above said amount of Rs.7500/- to the complainant with interest @ 8% per annum from the date of filing of the
complaint till realization.
          Let a copy of this order be sent to the parties as per regulation 21 of the Consumer Protection Regulations.  Thereafter file
be consigned to record room.
Announced on 05.12.19.
[HON'BLE MS. REKHA RANI]
PRESIDENT
[ KIRAN KAUSHAL]
MEMBER"""
   summary = summarize(text)
   print(f"Original Text:\n {text}")
   print(f"\nSummary:\n {summary}")