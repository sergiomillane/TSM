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
/// \file DetectorConstruction.cc
/// \brief Implementation of the B1::DetectorConstruction class

#include "DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4PhysicalConstants.hh"
#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Tubs.hh"
#include "G4Material.hh"
#include "G4Orb.hh"
#include "G4Sphere.hh"
#include "G4Trd.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"

namespace B1
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // Get nist material manager
  G4NistManager* nist = G4NistManager::Instance();

  // Envelope parameters
  //
  G4double env_sizeXY = 300*cm, env_sizeZ = 300*cm;
  G4Material* env_mat = nist->FindOrBuildMaterial("G4_AIR");

  // Option to switch on/off checking of volumes overlaps
  //
  G4bool checkOverlaps = true;

  //
  // World
  //
  G4double world_sizeXY = 1.2*env_sizeXY;
  G4double world_sizeZ  = 1.2*env_sizeZ;
  G4Material* world_mat = nist->FindOrBuildMaterial("G4_AIR");

  G4Box* solidWorld =
    new G4Box("World",                       //its name
       0.5*world_sizeXY, 0.5*world_sizeXY, 0.5*world_sizeZ);     //its size

  G4LogicalVolume* logicWorld =
    new G4LogicalVolume(solidWorld,          //its solid
                        world_mat,           //its material
                        "World");            //its name

  G4VPhysicalVolume* physWorld =
    new G4PVPlacement(0,                     //no rotation
                      G4ThreeVector(),       //at (0,0,0)
                      logicWorld,            //its logical volume
                      "World",               //its name
                      0,                     //its mother  volume
                      false,                 //no boolean operation
                      0,                     //copy number
                      checkOverlaps);        //overlaps checking

  //
  // Envelope
  //
  G4Box* solidEnv =
    new G4Box("Envelope",                    //its name
        0.5*env_sizeXY, 0.5*env_sizeXY, 0.5*env_sizeZ); //its size

  G4LogicalVolume* logicEnv =
    new G4LogicalVolume(solidEnv,            //its solid
                        env_mat,             //its material
                        "Envelope");         //its name

  new G4PVPlacement(0,                       //no rotation
                    G4ThreeVector(),         //at (0,0,0)
                    logicEnv,                //its logical volume
                    "Envelope",              //its name
                    logicWorld,              //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking








 {
 G4Material* shape5_mat = nist->FindOrBuildMaterial("G4_W");
  G4ThreeVector pos5 = G4ThreeVector(0*cm, 45*cm, 0*cm);

  // Trapezoid shape
  G4Box* solidShape5 =
    new G4Box("TUNGSTENO",                       //its name
       50*cm,0.04*cm,50*cm);     //its size

  G4LogicalVolume* logicShape5 =
    new G4LogicalVolume(solidShape5,         //its solid
                        shape5_mat,          //its material
                        "TUNGSTENO");           //its name

  new G4PVPlacement(0,                       //no rotation
                    pos5,                    //at position
                    logicShape5,             //its logical volume
                    "Shape5",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape5 as scoring volume
  //

fScoringVolume5 = logicShape5;
 //
//G4Tubs *solidDetector2 = new G4Tubs("solidDetector2", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(0*cm, 0*cm, 0*cm);
//G4LogicalVolume *logicDetector2 = new G4LogicalVolume(solidDetector2, world_mat, "logicDetector2");
}
  //
  // Shape e
  //
   {
 G4Material* shapeE_mat = nist->FindOrBuildMaterial("G4_A-150_TISSUE");
  G4ThreeVector posE = G4ThreeVector(0*cm, -50*cm, 0*cm);

  // Trapezoid shape
  G4Sphere* solidShapeE =
    new G4Sphere("Esfera", 0*cm, 30*cm,
		0, 6.2831, 0, 6.2831);     //its size
G4Box* solidShapeX =
    new G4Box("ShapeX",                       //its name
       22.36*cm,0.1*cm,22.36*cm);     //its size

//G4SubtractionSolid *solidResta=G4SubtractionSolid ("Resta", solidShapeE, solidShapeX);

  G4LogicalVolume* logicShapeE =
    new G4LogicalVolume(solidShapeE,         //its solid
                        shapeE_mat,          //its material
                        "ShapeE");           //its name

  new G4PVPlacement(0,                       //no rotation
                    posE,                    //at position
                    logicShapeE,             //its logical volume
                    "ShapeE",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shapee as scoring volume
  //

fScoringVolumeE = logicShapeE;
 //
//G4Tubs *solidDetector2 = new G4Tubs("solidDetector2", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(0*cm, 0*cm, 0*cm);
//G4LogicalVolume *logicDetector2 = new G4LogicalVolume(solidDetector2, world_mat, "logicDetector2");

G4Material* shapeX_mat = nist->FindOrBuildMaterial("G4_Galactic");
  G4ThreeVector posX = G4ThreeVector(0*cm, -30*cm, 0*cm);



  // Conical section shape
  


  G4LogicalVolume* logicShapeX =
    new G4LogicalVolume(solidShapeX,         //its solid
                        shapeX_mat,          //its material
                        "ShapeX");           //its name

  new G4PVPlacement(0,                       //no rotation
                    posX,                    //at position
                    logicShapeX,             //its logical volume
                    "ShapeX",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape4 as scoring volume
  //
fScoringVolumeX = logicShapeX;


}



 {
 G4Material* shape5_mat = nist->FindOrBuildMaterial("G4_W");
  G4ThreeVector pos5 = G4ThreeVector(15*cm, 0*cm, 5*cm);

  // Trapezoid shape
  G4Box* solidShape5 =
    new G4Box("Shape5",                       //its name
       10*cm,4*cm,10*cm);     //its size

  G4LogicalVolume* logicShape5 =
    new G4LogicalVolume(solidShape5,         //its solid
                        shape5_mat,          //its material
                        "Shape5");           //its name

  new G4PVPlacement(0,                       //no rotation
                    pos5,                    //at position
                    logicShape5,             //its logical volume
                    "Shape5",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape5 as scoring volume
  //

fScoringVolume5 = logicShape5;
 //
//G4Tubs *solidDetector2 = new G4Tubs("solidDetector2", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(0*cm, 0*cm, 0*cm);
//G4LogicalVolume *logicDetector2 = new G4LogicalVolume(solidDetector2, world_mat, "logicDetector2");
}

  // Shape 2
  //

{
 G4Material* shape2_mat = nist->FindOrBuildMaterial("G4_W");
  G4ThreeVector pos2 = G4ThreeVector(15*cm, 0*cm,-5*cm);

  // Trapezoid shape
 G4Box* solidShape2 =
    new G4Box("Shape2",                       //its name
       10*cm,4*cm,10*cm);     //its size

  G4LogicalVolume* logicShape2 =
    new G4LogicalVolume(solidShape2,         //its solid
                        shape2_mat,          //its material
                        "Shape2");           //its name

  new G4PVPlacement(0,                       //no rotation
                    pos2,                    //at position
                    logicShape2,             //its logical volume
                    "Shape2",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape2 as scoring volume
  //

fScoringVolume2 = logicShape2;
 //
//G4Tubs *solidDetector2 = new G4Tubs("solidDetector2", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(0*cm, 0*cm, 0*cm);
//G4LogicalVolume *logicDetector2 = new G4LogicalVolume(solidDetector2, world_mat, "logicDetector2");
}


// Shape 3

 {
  G4Material* shape3_mat = nist->FindOrBuildMaterial("G4_W");
  G4ThreeVector pos3 = G4ThreeVector(-15*cm, 0*cm, 5*cm);

  G4Box* solidShape3 =
    new G4Box("Shape3",                       //its name
       10*cm,4*cm,10*cm);     //its size

  G4LogicalVolume* logicShape3 =
    new G4LogicalVolume(solidShape3,          //its solid
                        shape3_mat,           //its material
                        "Shape3");            //its name

  new G4PVPlacement(0,                       //no rotation
                    pos3,                    //at position
                    logicShape3,             //its logical volume
                    "Shape3",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape3 as scoring volume3
  //

fScoringVolume3 = logicShape3;
//
//G4Tubs *solidDetector3 = new G4Tubs("solidDetector3", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(3*cm, 0*cm, 0*cm);
//G4LogicalVolume *logicDetector = new G4LogicalVolume(solidDetector3, world_mat, "logicDetector3");

}



// Shape 4

{
G4Material* shape4_mat = nist->FindOrBuildMaterial("G4_W");
  G4ThreeVector pos4 = G4ThreeVector(-15*cm, 0*cm, -5*cm);



  // Conical section shape
  G4Box* solidShape4 =
    new G4Box("Shape4",                       //its name
       10*cm,4*cm,10*cm);     //its size


  G4LogicalVolume* logicShape4 =
    new G4LogicalVolume(solidShape4,         //its solid
                        shape4_mat,          //its material
                        "Shape4");           //its name

  new G4PVPlacement(0,                       //no rotation
                    pos4,                    //at position
                    logicShape4,             //its logical volume
                    "Shape4",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape4 as scoring volume
  //
fScoringVolume4 = logicShape4;

  //
 //G4Tubs *solidDetector4 = new G4Tubs("solidDetector4", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(0*cm, -5*cm, 0*cm);
//G4LogicalVolume *logicDetector = new G4LogicalVolume(solidDetector4, world_mat, "logicDetector4");
//G4LogicalVolume logicDetector(solidDetector, worldMat, "logicDetector");
}


{
 G4Material* shapeA_mat = nist->FindOrBuildMaterial("G4_W");
  G4ThreeVector posA = G4ThreeVector(0*cm, 0*cm, -15*cm);

  // Trapezoid shape
  G4Box* solidShapeA =
    new G4Box("ShapeA",                       //its name
       10*cm,4*cm,10*cm);     //its size

  G4LogicalVolume* logicShapeA =
    new G4LogicalVolume(solidShapeA,         //its solid
                        shapeA_mat,          //its material
                        "ShapeA");           //its name

  new G4PVPlacement(0,                       //no rotation
                    posA,                    //at position
                    logicShapeA,             //its logical volume
                    "ShapeA",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape5 as scoring volume
  //

fScoringVolumeA = logicShapeA;
 //
//G4Tubs *solidDetector2 = new G4Tubs("solidDetector2", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(0*cm, 0*cm, 0*cm);
//G4LogicalVolume *logicDetector2 = new G4LogicalVolume(solidDetector2, world_mat, "logicDetector2");
}

  // Shape 2
  //

{
 G4Material* shapeB_mat = nist->FindOrBuildMaterial("G4_W");
  G4ThreeVector posB = G4ThreeVector(0*cm, 0*cm, 15*cm);

  // Trapezoid shape
 G4Box* solidShapeB =
    new G4Box("ShapeB",                       //its name
       10*cm,4*cm,10*cm);     //its size

  G4LogicalVolume* logicShapeB =
    new G4LogicalVolume(solidShapeB,         //its solid
                        shapeB_mat,          //its material
                        "ShapeB");           //its name

  new G4PVPlacement(0,                       //no rotation
                    posB,                    //at position
                    logicShapeB,             //its logical volume
                    "ShapeB",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape2 as scoring volume
  //

fScoringVolumeB = logicShapeB;
 //
//G4Tubs *solidDetector2 = new G4Tubs("solidDetector2", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(0*cm, 0*cm, 0*cm);
//G4LogicalVolume *logicDetector2 = new G4LogicalVolume(solidDetector2, world_mat, "logicDetector2");
}




// Shape 3

 {
  G4Material* shapeC_mat = nist->FindOrBuildMaterial("G4_Galactic");
  G4ThreeVector posC = G4ThreeVector(0*cm, -4*cm, 0*cm);

  G4Box* solidShapeC =
    new G4Box("ShapeC",                       //its name
       5*cm,0.1*cm,5*cm);     //its size

  G4LogicalVolume* logicShapeC =
    new G4LogicalVolume(solidShapeC,          //its solid
                        shapeC_mat,           //its material
                        "DetPostColimador");            //its name

  new G4PVPlacement(0,                       //no rotation
                    posC,                    //at position
                    logicShapeC,             //its logical volume
                    "DetPostColimador",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape3 as scoring volume3
  //

fScoringVolumeC = logicShapeC;
//
//G4Tubs *solidDetector3 = new G4Tubs("solidDetector3", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(3*cm, 0*cm, 0*cm);
//G4LogicalVolume *logicDetector = new G4LogicalVolume(solidDetector3, world_mat, "logicDetector3");

}
 



// Shape 4
/*
{
G4Material* shapeD_mat = nist->FindOrBuildMaterial("G4_W");
  G4ThreeVector posD = G4ThreeVector(-45*cm, 0*cm, -35*cm);



  // Conical section shape
  G4Box* solidShapeD =
    new G4Box("ShapeD",                       //its name
       10*cm,0.2*cm,30*cm);     //its size


  G4LogicalVolume* logicShapeD =
    new G4LogicalVolume(solidShapeD,         //its solid
                        shapeD_mat,          //its material
                        "Shape4");           //its name

  new G4PVPlacement(0,                       //no rotation
                    posD,                    //at position
                    logicShapeD,             //its logical volume
                    "ShapeD",                //its name
                    logicEnv,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  // Set Shape4 as scoring volume
  //
fScoringVolumeD = logicShapeD;








  //
 //G4Tubs *solidDetector4 = new G4Tubs("solidDetector4", 0*cm, 1*cm, 10*cm,1*cm,1*cm);
//G4ThreeVector pos5 = G4ThreeVector(0*cm, -5*cm, 0*cm);
//G4LogicalVolume *logicDetector = new G4LogicalVolume(solidDetector4, world_mat, "logicDetector4");
//G4LogicalVolume logicDetector(solidDetector, worldMat, "logicDetector");
}
*/







  //always return the physical World
  //
  return physWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}
