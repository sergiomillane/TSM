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
/// \file DetectorConstruction.hh
/// \brief Definition of the B1::DetectorConstruction class

#ifndef CONSTRUCTION_HH
#define CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4SubtractionSolid.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "globals.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;

/// Detector construction class to define materials and geometry.

namespace B1
{

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;

    
    G4LogicalVolume* GetScoringVolume2() const { return fScoringVolume2; }
    G4LogicalVolume* GetScoringVolume3() const { return fScoringVolume3; }
    G4LogicalVolume* GetScoringVolume4() const { return fScoringVolume4; }
    G4LogicalVolume* GetScoringVolume5() const { return fScoringVolume5; }
    G4LogicalVolume* GetScoringVolumeA() const { return fScoringVolumeA; }
    G4LogicalVolume* GetScoringVolumeB() const { return fScoringVolumeB; }
    G4LogicalVolume* GetScoringVolumeC() const { return fScoringVolumeC; }
    G4LogicalVolume* GetScoringVolumeD() const { return fScoringVolumeD; }
    G4LogicalVolume* GetScoringVolumeE() const { return fScoringVolumeE; }
    G4LogicalVolume* GetScoringVolumeX() const { return fScoringVolumeX; }
  protected:
    
    G4LogicalVolume* fScoringVolume2 = nullptr;
    G4LogicalVolume* fScoringVolume3 = nullptr;
    G4LogicalVolume* fScoringVolume4 = nullptr;
    G4LogicalVolume* fScoringVolume5 = nullptr;
    G4LogicalVolume* fScoringVolumeA = nullptr;
    G4LogicalVolume* fScoringVolumeB = nullptr;
    G4LogicalVolume* fScoringVolumeC = nullptr;
    G4LogicalVolume* fScoringVolumeD = nullptr;
    G4LogicalVolume* fScoringVolumeE = nullptr;
    G4LogicalVolume* fScoringVolumeX = nullptr;
};

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
