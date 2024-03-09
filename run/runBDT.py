import logging
logging.basicConfig(level=logging.INFO)

import ROOT
from ROOT import TMVA
from ROOT import TFile, TTree, TCut, TSystem, TObjString, TH1, TStyle, TBranch, TChain, TCanvas, TLegend, TString, TMath, TH1F
from ROOT import gStyle, gROOT
from ROOT import kTRUE
from subprocess import call
from os.path import isfile
import torch
import torch.optim as optim
from torch import nn
from torchmetrics import Accuracy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gStyle.SetOptTitle(0)
gStyle.SetOptStat(0)
gStyle.SetStatBorderSize(1)
gStyle.SetStatX(.50)
gStyle.SetStatY(.30)
gROOT.ForceStyle()

TH1.SetDefaultSumw2(kTRUE)

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

outputFile  = TFile.Open('TMVA_BDT.root', 'RECREATE')
factory     = TMVA.Factory('BDT_classification', outputFile,'!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')
dataloader1  = TMVA.DataLoader("BDT_classification")
factory.SetVerbose()

inbkg = TFile.Open("/eos/home-i/ibessudo/Boosted_ditau_ML/ttbar/forbdt_fixed.root")
insig = TFile.Open("/eos/home-i/ibessudo/Boosted_ditau_ML/ttx/forbdt_fixed.root")

sigtree = insig.Get("forbdt")
bkgtree = inbkg.Get("forbdt")

real_var_alias_dict = {"pt":"ditau_pt","pt_leadSubjets":"ditau_pt_leadSubjets", "eta":"ditau_eta"}
n_var_alias_dict    = {"n_isotrack":"ditau_n_isotrack", "n_othertrack":"ditau_n_othertrack",
                        "n_track":"ditau_n_track", "n_iso_ellipse":"ditau_n_iso_ellipse",
                        "n_tracks_subl":"ditau_n_tracks_subl", "n_tracks_lead":"ditau_n_tracks_lead",
                        "n_subjets":"ditau_n_subjets"}
f_var_alias_dict    = {"f_subjets":"ditau_f_subjets", "f_subjet_lead":"ditau_f_subjet_lead",
                        "f_subjet_subl":"ditau_f_subjet_subl", "f_track_lead":"ditau_f_track_lead",
                        "f_track_subl":"ditau_f_track_subl", "f_isotracks":"ditau_f_isotracks"}
R_var_alias_dict    = {"R_max_lead":"ditau_R_max_lead", "R_max_subl":"ditau_R_max_subl",
                        "R_core_lead":"ditau_R_core_lead", "R_core_subl":"ditau_R_core_subl",
                        "R_tracks_lead":"ditau_R_tracks_lead", "R_tracks_subl":"ditau_R_tracks_subl",
                        "R_subjets_subl":"ditau_R_subjets_subl", "R_subjets_subsubl":"ditau_R_subjets_subsubl",
                        "R_isotrack":"ditau_R_isotrack", "R_track":"ditau_R_track", "R_track_core":"ditau_R_track_core",
                        "R_track_all":"ditau_R_track_all"}
m_var_alias_dict    = {"m_track":"ditau_m_track","m_track_core":"ditau_m_track_core",
                        "m_track_all":"ditau_m_track_all","m_core_lead":"ditau_m_core_lead",
                        "m_core_subl":"ditau_m_core_subl","m_tracks_lead":"ditau_m_tracks_lead",
                        "m_tracks_subl":"ditau_m_tracks_subl"}
E_var_alias_dict    = {"E_frac_subl":"ditau_E_frac_subl", "E_frac_subsubl":"ditau_E_frac_subsubl"}
d0_var_alias_dict   = {"d0_leadtrack_subl":"ditau_d0_leadtrack_subl","d0_leadtrack_lead":"ditau_d0_leadtrack_lead"}

alias_dict = real_var_alias_dict | n_var_alias_dict | f_var_alias_dict | R_var_alias_dict | m_var_alias_dict | E_var_alias_dict | d0_var_alias_dict

for k,v in alias_dict.items():
    sigtree.SetAlias(k,v)
    bkgtree.SetAlias(k,v)


varialbes_dict = {"f_subjet_subl":"f_{subjet}^{subl}","f_subjets":"f_{subjets}",
                    "f_track_lead":"f_{track}^{lead}","f_track_subl":"f_{track}^{subl}",
                    "R_max_lead":"R_{max}^{lead}","R_max_subl":"R_{max}^{subl}",
                    "R_isotrack":"R_{isotrack}","R_track":"R_{track}","R_core_lead":"R_{core}^{lead}",
                    "R_core_subl":"R_{core}^{subl}","log(m_tracks_lead)":"log(m_{tracks}^{lead})",
                    "log(m_tracks_subl)":"log(m_{tracks}^{subl})","abs(d0_leadtrack_subl)":"|d0_{leadtrk}^{subl}|",
                    "abs(d0_leadtrack_lead)":"|d0_{leadtrk}^{lead}|"}
for k,v in varialbes_dict.items():
    dataloader1.AddVariable(k,v,"", 'F')

dataloader1.AddBackgroundTree(bkgtree, 1.0)
dataloader1.AddSignalTree(sigtree, 1.0)

dataloader1.PrepareTrainingAndTestTree(TCut(''),"SplitMode=random:!V::TrainTestSplit_Background=0.75:TrainTestSplit_Signal=0.75")

factory.BookMethod( dataloader1, TMVA.Types.kBDT, "BDT1",  "!H:!V:NTrees=1000:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.2:UseBaggedBoost:BaggedSampleFraction=0.3:SeparationType=GiniIndex:nCuts=300:MinNodeSize=0.5%" )

logging.info('training methods')
factory.TrainAllMethods()

logging.info('testing methods')
factory.TestAllMethods()

logging.info('evaluating methods')
factory.EvaluateAllMethods()

# count num. of params

bdt_method = factory.GetMethod("BDT_classification","BDT1")

forest_vector = bdt_method.GetForest()

n_nodes_per_tree = []

for t in forest_vector:
    n_nodes_per_tree.append(t.CountLeafNodes(t.GetRoot()) - 1)
    
tot_n_nodes = np.sum(n_nodes_per_tree)

n_variables = len(varialbes_dict)

n_params = tot_n_nodes * n_variables

print(f'Total number of parameters for BDT is the number of nodes times the number of variables, which is {n_params}')




c_resp = TCanvas("plotcanvas", "plotcanvas", 1200,1000)
c_resp.cd()

g1 = factory.GetROCCurve(dataloader1,"BDT1")
print("number of bin is {}".format(g1.GetN()))

with open("ROC1.csv","w") as f:
    f.write("j,xpoint,1/1-ypoint\n")
    for j in range(g1.GetN()):
        xpoint = np.double(0.0)
        ypoint = np.double(0.0)
        g1.GetPoint(j,xpoint,ypoint)
        if g1.GetPoint(j,xpoint,ypoint) != -1:
            if ypoint != 1:
                g1.SetPoint(j,xpoint,1./(1-ypoint))
                f.write('{},{},{}\n'.format(j,xpoint,1./(1-ypoint)))
g1.SetMarkerColor(2)
g1.SetMinimum(1.)

g1.SetMarkerStyle(20)
g1.SetMarkerSize(0.5)
legend=TLegend(0.7,0.7,0.9,0.8)
legend.SetBorderSize(0)
legend.AddEntry(g1,"Var1 - |d0|","p")
c_resp.cd()
c_resp.SetLogy()
c_resp.Update()
g1.Draw("Psame")
legend.Draw("same")

c_resp.SaveAs("ROC.png")

logging.info("Finished running everything")

outputFile.Close()

del factory
