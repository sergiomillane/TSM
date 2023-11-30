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
/// \file SteppingAction.cc
/// \brief Implementation of the B1::SteppingAction class

#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"

#include "G4Step.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4LogicalVolume.hh"

namespace B1
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(EventAction* eventAction)
: fEventAction(eventAction)
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* step)
{

 int i=0;
  //G4cout<<"testi v2=  "<<fScoringVolume2<<" "<<i<<" R3 v3="<<fScoringVolume3<<" "<<" R4 v4="<<fScoringVolume4<<" "<<" R5 v5="<<fScoringVolume5<<G4endl;
  if (!fScoringVolume2) {
//G4cout<<i<<" R1 = xxxxxxx"<<!fScoringVolume2<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolume2 = detConstruction->GetScoringVolume2();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }


  if (!fScoringVolume3) {
//G4cout<<i<<" R1 ="<<!fScoringVolume3<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolume3 = detConstruction->GetScoringVolume3();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }

if (!fScoringVolume4) {
//G4cout<<i<<" R1 ="<<!fScoringVolume<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolume4 = detConstruction->GetScoringVolume4();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }

if (!fScoringVolume5) {
//G4cout<<i<<" R1 ="<<!fScoringVolume<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolume5 = detConstruction->GetScoringVolume5();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }

  if (!fScoringVolumeA) {
//G4cout<<i<<" R1 = xxxxxxx"<<!fScoringVolume2<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolumeA = detConstruction->GetScoringVolumeA();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }


  if (!fScoringVolumeB) {
//G4cout<<i<<" R1 ="<<!fScoringVolume3<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolumeB = detConstruction->GetScoringVolumeB();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }

if (!fScoringVolumeC) {
//G4cout<<i<<" R1 ="<<!fScoringVolume<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolumeC = detConstruction->GetScoringVolumeC();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }

if (!fScoringVolumeD) {
//G4cout<<i<<" R1 ="<<!fScoringVolume<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolumeD = detConstruction->GetScoringVolumeD();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }

  if (!fScoringVolumeE) {
//G4cout<<i<<" R1 ="<<!fScoringVolume<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolumeE = detConstruction->GetScoringVolumeE();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }

  if (!fScoringVolumeX) {
//G4cout<<i<<" R1 ="<<!fScoringVolume<<G4endl;
    const DetectorConstruction* detConstruction
      = static_cast<const DetectorConstruction*>
        (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolumeX = detConstruction->GetScoringVolumeX();
   //G4cout<<"testi =  "<<fScoringVolume<<G4endl;
  i++;
  }

//G4cout<<i<<" R2 ="<<!fScoringVolume<<G4endl;
  // get volume of the current step
  G4LogicalVolume* volume
    = step->GetPreStepPoint()->GetTouchableHandle()
      ->GetVolume()->GetLogicalVolume();

  // check if we are in scoring volume
 //G4cout<<"p2 v2=  "<<fScoringVolume2<<" "<<i<<" R3 v3="<<fScoringVolume3<<" "<<" R4 v4="<<fScoringVolume4<<" "<<" R5 v5="<<fScoringVolume5<<" Rx="<<volume<<G4endl;
   if(volume != fScoringVolume2) {
    if (volume != fScoringVolume3){
     if (volume != fScoringVolume4){
      if (volume != fScoringVolume5){
      if(volume != fScoringVolumeA) {
    if (volume != fScoringVolumeB){
     if (volume != fScoringVolumeC){
      if (volume != fScoringVolumeD){
      if (volume != fScoringVolumeE){
      if (volume != fScoringVolumeX){
 //G4cout<<"p3 v2=  "<<fScoringVolume2<<" "<<i<<" R3 v3="<<fScoringVolume3<<" "<<" R4 v4="<<fScoringVolume4<<" "<<" R5 v5="<<fScoringVolume5<<G4endl;
	//return;
}
}
}
}
}
}
}
}
}
}
//G4cout<<volume <<" R3 ="<<fScoringVolume<<" vTT "<<fScoringVolume2<<G4endl;

//G4cout<<i<<" R1 ="<<volume<<"  "<<fScoringVolume2<<G4endl;
 if(volume == fScoringVolume2) {
//G4cout<<i<<" R1 ="<<volume<<G4endl;
      //G4double edepStep = step->GetTotalEnergyDeposit();
      //fEventAction->AddEdep(edepStep);
	G4double twoedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddtwoEdep(twoedepStep);

				}

 if(volume == fScoringVolume3) {
      //G4double edepStep = step->GetTotalEnergyDeposit();
      //fEventAction->AddEdep(edepStep);
	G4double tresedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddtresEdep(tresedepStep);
                               }

if(volume == fScoringVolume4) {
      //G4double edepStep = step->GetTotalEnergyDeposit();
      //fEventAction->AddEdep(edepStep);
	G4double cuatroedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddcuatroEdep(cuatroedepStep);
                               }


if(volume == fScoringVolume5) {
  // G4cout<<volume <<" R4"<<G4endl;
  // collect energy deposited in this step
  G4double cincoedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddcincoEdep(cincoedepStep);
			}

if(volume == fScoringVolumeA) {
//G4cout<<i<<" R1 ="<<volume<<G4endl;
      //G4double edepStep = step->GetTotalEnergyDeposit();
      //fEventAction->AddEdep(edepStep);
	G4double AedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddAEdep(AedepStep);

				}

 if(volume == fScoringVolumeB) {
      //G4double edepStep = step->GetTotalEnergyDeposit();
      //fEventAction->AddEdep(edepStep);
	G4double BedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddBEdep(BedepStep);
                               }

if(volume == fScoringVolumeC) {
G4Track *track=step->GetTrack();
G4ThreeVector pos1=track->GetPosition();
G4ThreeVector mom1=track->GetMomentum();

G4String particoli=track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName(); 
  // G4cout<<volume <<" R4"<<G4endl;
  // collect energy deposited in this step
if(particoli=="gamma"){ 
FILE   *fp3_1 = fopen("ColiDet","a+"); 
			fprintf(fp3_1, "%f %f %f %f %f %f  \n", pos1[0],mom1[0],pos1[1],mom1[1],pos1[2],mom1[2]);
			fclose(fp3_1);
} 
      //G4double edepStep = step->GetTotalEnergyDeposit();
      //fEventAction->AddEdep(edepStep);
	G4double CedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddCEdep(CedepStep);
                               }


if(volume == fScoringVolumeD) {
  // G4cout<<volume <<" R4"<<G4endl;
  // collect energy deposited in this step
  G4double DedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddDEdep(DedepStep);
			}

if(volume == fScoringVolumeE) {
  // G4cout<<volume <<" R4"<<G4endl;
  // collect energy deposited in this step
  G4double EedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddEEdep(EedepStep);
			}



if(volume == fScoringVolumeX) {
G4Track *track=step->GetTrack();
G4ThreeVector pos=track->GetPosition();
G4ThreeVector mom=track->GetMomentum();

G4String parti=track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName(); 
  // G4cout<<volume <<" R4"<<G4endl;
  // collect energy deposited in this step
if(parti=="gamma"){ 
FILE   *fp3_1 = fopen("DatosTSM","a+"); 
			fprintf(fp3_1, "%f %f %f %f %f %f  \n", pos[0],mom[0],pos[1],mom[1],pos[2],mom[2] );
			fclose(fp3_1);
} 
  G4double XedepStep = step->GetTotalEnergyDeposit();
  fEventAction->AddXEdep(XedepStep);
  	//G4cout<< pos<< " X  " <<mom  << "  "  << " PRUEBA2 " <<step->GetTrack()->GetCurrentStepNumber() <<G4endl;
			}

if (step->GetTrack()->GetCurrentStepNumber()==2){
	G4Track *track=step->GetTrack();
G4ThreeVector pos1=track->GetPosition();
G4ThreeVector mom1=track->GetMomentum();
G4double E1=track->GetKineticEnergy();
	
	}
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
}

