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
/// \file RunAction.hh
/// \brief Definition of the B1::RunAction class

#ifndef B1RunAction_h
#define B1RunAction_h 1

#include "G4UserRunAction.hh"
#include "G4Accumulable.hh"
#include "globals.hh"

class G4Run;

/// Run action class
///
/// In EndOfRunAction(), it calculates the dose in the selected volume
/// from the energy deposit accumulated via stepping and event actions.
/// The computed dose is then printed on the screen.

namespace B1
{

class RunAction : public G4UserRunAction
{
  public:
    RunAction();
    ~RunAction() override;

    void BeginOfRunAction(const G4Run*) override;
    void   EndOfRunAction(const G4Run*) override;


    void AddtwoEdep (G4double twoedep);
    void AddtresEdep (G4double tresedep);
    void AddcuatroEdep (G4double cuatroedep);
    void AddcincoEdep (G4double cincoedep);

    void AddAEdep (G4double Aedep);
    void AddBEdep (G4double Bedep);
    void AddCEdep (G4double Cedep);
    void AddDEdep (G4double Dedep);

    void AddEEdep (G4double Eedep);
    void AddXEdep (G4double Xedep);
  private:

    G4Accumulable<G4double> ftwoEdep = 0.;
    G4Accumulable<G4double> ftwoEdep2 = 0.;
    G4Accumulable<G4double> ftresEdep = 0.;
    G4Accumulable<G4double> ftresEdep2 = 0.;
    G4Accumulable<G4double> fcuatroEdep = 0.;
    G4Accumulable<G4double> fcuatroEdep2 = 0.;
    G4Accumulable<G4double> fcincoEdep = 0.;
    G4Accumulable<G4double> fcincoEdep2 = 0.;

    G4Accumulable<G4double> fAEdep = 0.;
    G4Accumulable<G4double> fAEdep2 = 0.;
    G4Accumulable<G4double> fBEdep = 0.;
    G4Accumulable<G4double> fBEdep2 = 0.;
    G4Accumulable<G4double> fCEdep = 0.;
    G4Accumulable<G4double> fCEdep2 = 0.;
    G4Accumulable<G4double> fDEdep = 0.;
    G4Accumulable<G4double> fDEdep2 = 0.;
    G4Accumulable<G4double> fEEdep = 0.;
    G4Accumulable<G4double> fEEdep2 = 0.;
    G4Accumulable<G4double> fXEdep = 0.;
    G4Accumulable<G4double> fXEdep2 = 0.;




};

}

#endif

