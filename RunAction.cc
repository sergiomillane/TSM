//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file RunAction.cc
/// \brief Implementation of the B1::RunAction class

#include "RunAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "DetectorConstruction.hh"
// #include "Run.hh"

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4AccumulableManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4UnitsTable.hh"
#include <stdio.h>
#include "G4SystemOfUnits.hh"


using namespace std;
namespace B1
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction()
{
 FILE *fp;
    fp = fopen("datos.txt","w+"); 
fprintf(fp, " ");
fclose(fp);

 FILE *fp2;
    fp2 = fopen("datos2.txt","w+"); 
fprintf(fp2, " ");
fclose(fp2);
  // add new units for dose
  //
  const G4double milligray = 1.e-3*gray;
  const G4double microgray = 1.e-6*gray;
  const G4double nanogray  = 1.e-9*gray;
  const G4double picogray  = 1.e-12*gray;

  new G4UnitDefinition("milligray", "milliGy" , "Dose", milligray);
  new G4UnitDefinition("microgray", "microGy" , "Dose", microgray);
  new G4UnitDefinition("nanogray" , "nanoGy"  , "Dose", nanogray);
  new G4UnitDefinition("picogray" , "picoGy"  , "Dose", picogray);

  // Register accumulable to the accumulable manager
  G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();

  accumulableManager->RegisterAccumulable(ftwoEdep);
  accumulableManager->RegisterAccumulable(ftwoEdep2);
 accumulableManager->RegisterAccumulable(ftresEdep);
  accumulableManager->RegisterAccumulable(ftresEdep2);
accumulableManager->RegisterAccumulable(fcuatroEdep);
  accumulableManager->RegisterAccumulable(fcuatroEdep2);
accumulableManager->RegisterAccumulable(fcincoEdep);
  accumulableManager->RegisterAccumulable(fcincoEdep2);
    accumulableManager->RegisterAccumulable(fAEdep);
  accumulableManager->RegisterAccumulable(fAEdep2);
 accumulableManager->RegisterAccumulable(fBEdep);
  accumulableManager->RegisterAccumulable(fBEdep2);
accumulableManager->RegisterAccumulable(fCEdep);
  accumulableManager->RegisterAccumulable(fCEdep2);
accumulableManager->RegisterAccumulable(fDEdep);
  accumulableManager->RegisterAccumulable(fDEdep2);
accumulableManager->RegisterAccumulable(fEEdep);
  accumulableManager->RegisterAccumulable(fEEdep2);
accumulableManager->RegisterAccumulable(fXEdep);
  accumulableManager->RegisterAccumulable(fXEdep2);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::~RunAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run*)
{
  // inform the runManager to save random number seed
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);

  // reset accumulables to their initial values
  G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Reset();

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run* run)
{
const G4double picogray  = 1.e-12*gray;
new G4UnitDefinition("picogray" , "picoGy"  , "Dose", picogray);
  G4int nofEvents = run->GetNumberOfEvent();
  if (nofEvents == 0) return;

  // Merge accumulables
  G4AccumulableManager* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Merge();

  // Compute dose = total energy deposit in a run and its variance
  //

  G4double twoedep  = ftwoEdep.GetValue();
  G4double twoedep2 = ftwoEdep2.GetValue();
  G4double tresedep  = ftresEdep.GetValue();
  G4double tresedep2 = ftresEdep2.GetValue();
  G4double cuatroedep  = fcuatroEdep.GetValue();
  G4double cuatroedep2 = fcuatroEdep2.GetValue();
  G4double cincoedep  = fcincoEdep.GetValue();
  G4double cincoedep2 = fcincoEdep2.GetValue();

  G4double Aedep  = fAEdep.GetValue();
  G4double Aedep2 = fAEdep2.GetValue();
  G4double Bedep  = fBEdep.GetValue();
  G4double Bedep2 = fBEdep2.GetValue();
  G4double Cedep  = fCEdep.GetValue();
  G4double Cedep2 = fCEdep2.GetValue();
  G4double Dedep  = fDEdep.GetValue();
  G4double Dedep2 = fDEdep2.GetValue();

 G4double Eedep  = fEEdep.GetValue();
  G4double Eedep2 = fEEdep2.GetValue();

 G4double Xedep  = fXEdep.GetValue();
  G4double Xedep2 = fXEdep2.GetValue();



 G4double tworms = twoedep2 - twoedep*twoedep/nofEvents;
  if (tworms > 0.) tworms = std::sqrt(tworms); else tworms = 0.;

 G4double tresrms = tresedep2 - tresedep*tresedep/nofEvents;
  if (tresrms > 0.) tresrms = std::sqrt(tresrms); else tresrms = 0.;

 G4double cuatrorms = cuatroedep2 - cuatroedep*cuatroedep/nofEvents;
  if (cuatrorms > 0.) cuatrorms = std::sqrt(cuatrorms); else cuatrorms = 0.;

 G4double cincorms = cincoedep2 - cincoedep*cincoedep/nofEvents;
  if (cincorms > 0.) cincorms = std::sqrt(cincorms); else cincorms = 0.;

   G4double Arms = Aedep2 - Aedep*Aedep/nofEvents;
  if (Arms > 0.) Arms = std::sqrt(Arms); else Arms = 0.;

 G4double Brms = Bedep2 - Bedep*Bedep/nofEvents;
  if (Brms > 0.) Brms = std::sqrt(Brms); else Brms = 0.;

 G4double Crms = Cedep2 - Cedep*Cedep/nofEvents;
  if (Crms > 0.) Crms = std::sqrt(Crms); else Crms = 0.;

 G4double Drms = Dedep2 - Dedep*Dedep/nofEvents;
  if (Drms > 0.) Drms = std::sqrt(Drms); else Drms = 0.;

G4double Erms = Eedep2 - Eedep*Eedep/nofEvents;
  if (Erms > 0.) Erms = std::sqrt(Erms); else Erms = 0.;

G4double Xrms = Xedep2 - Xedep*Xedep/nofEvents;
  if (Xrms > 0.) Xrms = std::sqrt(Xrms); else Xrms = 0.;

  const DetectorConstruction* detConstruction
   = static_cast<const DetectorConstruction*>
     (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
  G4double mass = detConstruction->GetScoringVolume5()->GetMass();


 G4double twodose = twoedep/mass;
 G4double twormsDose = tworms/mass;

G4double tresdose = tresedep/mass;
 G4double tresrmsDose = tresrms/mass;

G4double cuatrodose = cuatroedep/mass;
 G4double cuatrormsDose = cuatrorms/mass;

G4double cincodose = cincoedep/mass;
 G4double cincormsDose = cincorms/mass;


 G4double Adose = Aedep/mass;
 G4double ArmsDose = Arms/mass;

G4double Bdose = Bedep/mass;
 G4double BrmsDose = Brms/mass;

G4double Cdose = Cedep/mass;
 G4double CrmsDose = Crms/mass;

G4double Ddose = Dedep/mass;
 G4double DrmsDose = Drms/mass;

G4double Edose = Eedep/mass;
 G4double ErmsDose = Erms/mass;

G4double Xdose = Xedep/mass;
 G4double XrmsDose = Xrms/mass;


  // Run conditions
  //  note: There is no primary generator action object for "master"
  //        run manager for multi-threaded mode.
  const PrimaryGeneratorAction* generatorAction
   = static_cast<const PrimaryGeneratorAction*>
     (G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction());
  G4String runCondition;
  if (generatorAction)
  {
    const G4ParticleGun* particleGun = generatorAction->GetParticleGun();
    runCondition += particleGun->GetParticleDefinition()->GetParticleName();
    runCondition += " of ";
    G4double particleEnergy = particleGun->GetParticleEnergy();
    runCondition += G4BestUnit(particleEnergy,"Energy");
  }


  // Print
  //
  if (IsMaster()) {
    G4cout
     << G4endl
     << "--------------------End of Global Run-----------------------";
  }
  else {
    G4cout
     << G4endl
     << "--------------------End of Local Run------------------------";
  }



   FILE *pFile ;

   pFile = fopen ("output.txt","a+");
  //fprintf (pFile, "vol1 vol2 vol3 vol4   \n");

	 

  G4cout
     << G4endl
     << " The run consists of " << nofEvents << " "<< runCondition
     << G4endl
     << "Cumulated dose per run, in scoring volume1 : "
     << G4BestUnit(cincodose,"Dose") << "rms=" << G4BestUnit(cincormsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in scoring volume2 : "
     << G4BestUnit(twodose,"Dose") << "rms=" << G4BestUnit(twormsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in scoring volume3 : "
     << G4BestUnit(tresdose,"Dose") << "rms=" << G4BestUnit(tresrmsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in scoring volume4 : "
     << G4BestUnit(cuatrodose,"Dose") << "rms=" << G4BestUnit(cuatrormsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in scoring volumeA : "
     << G4BestUnit(Adose,"Dose") << "rms=" << G4BestUnit(ArmsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in scoring volumeB : "
     << G4BestUnit(Bdose,"Dose") << "rms=" << G4BestUnit(BrmsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in scoring volumeC: "
     << G4BestUnit(Cdose,"Dose") << "rms=" << G4BestUnit(CrmsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in scoring volumeD : "
     << G4BestUnit(Ddose,"Dose") << "rms=" << G4BestUnit(DrmsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in SPHERE : "
     << G4BestUnit(Edose,"Dose") << "rms=" << G4BestUnit(ErmsDose,"Dose")
     << G4endl
     << "Cumulated dose per run, in FINAL DETECTOR  : "
     << G4BestUnit(Xdose,"Dose") << "rms=" << G4BestUnit(XrmsDose,"Dose")
     << "------------------------------------------------------------"
     << G4endl
     << G4endl;

//G4cout<< "LLL"<<cincodose/picogray<<G4endl;
string pru=G4BestUnit(cincodose,"Dose");
     fprintf (pFile, "dosis %d  %e  %e  \n", nofEvents, Edose, Xdose);
     //fprintf (pFile, "%.12e %.12e %.12e %.12e\n",cincodose/picogray, twodose/picogray, tresdose/picogray, cuatrodose/picogray);

     fclose(pFile);


}   

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


void RunAction::AddtwoEdep(G4double twoedep){


  ftwoEdep  += twoedep;
  ftwoEdep2 += twoedep*twoedep;



}

void RunAction::AddtresEdep(G4double tresedep){


  ftresEdep  += tresedep;
  ftresEdep2 += tresedep*tresedep;


}

void RunAction::AddcuatroEdep(G4double cuatroedep){


  fcuatroEdep  += cuatroedep;
  fcuatroEdep2 += cuatroedep*cuatroedep;

}


void RunAction::AddcincoEdep(G4double cincoedep){


  fcincoEdep  += cincoedep;
  fcincoEdep2 += cincoedep*cincoedep;

}

void RunAction::AddAEdep(G4double Aedep){


  fAEdep  += Aedep;
  fAEdep2 += Aedep*Aedep;



}

void RunAction::AddBEdep(G4double Bedep){


  fBEdep  += Bedep;
  fBEdep2 += Bedep*Bedep;


}

void RunAction::AddCEdep(G4double Cedep){


  fCEdep  += Cedep;
  fCEdep2 += Cedep*Cedep;

}


void RunAction::AddDEdep(G4double Dedep){


  fDEdep  += Dedep;
  fDEdep2 += Dedep*Dedep;

}

void RunAction::AddEEdep(G4double Eedep){


  fEEdep  += Eedep;
  fEEdep2 += Eedep*Eedep;

}

void RunAction::AddXEdep(G4double Xedep){


  fXEdep  += Xedep;
  fXEdep2 += Xedep*Xedep;

}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}
