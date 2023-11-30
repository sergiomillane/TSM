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
/// \file PrimaryGeneratorAction.cc
/// \brief Implementation of the B1::PrimaryGeneratorAction class

#include "PrimaryGeneratorAction.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4RunManager.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include <CLHEP/Vector/ThreeVector.h>

namespace B1
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::PrimaryGeneratorAction()
{
  G4int n_particle = 1;
  fParticleGun  = new G4ParticleGun(n_particle);

  // default particle kinematic
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  G4ParticleDefinition* particle
    = particleTable->FindParticle(particleName="e-");
  fParticleGun->SetParticleDefinition(particle);
  //fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0,1,0));
  fParticleGun->SetParticleEnergy(5.74833*MeV);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void split_c(std::string str, std::vector<float> *v)
{
    std::string w = "";
    std::vector<float> dummy = {};
    for (auto x : str) 
    {
        if (x == '	')
        {
            // cout << w << endl;
            dummy.push_back(std::stof(w));
            w = "";
        }
        else if(x == '\"'){
            // dummy.push_back(std::stof(w));
            // w = "";
			continue;
		}
		else{
            w = w + x;
        }
    }
	dummy.push_back(std::stof(w));
    *v = dummy;
    // cout << w << endl;
}
/*
void PrimaryGeneratorAction::GeneratePrimariesFile(G4Event* anEvent)
{
  //this function is called at the begining of ecah event
  //

  // In order to avoid dependence of PrimaryGeneratorAction
  // on DetectorConstruction class we get Envelope volume
  // from G4LogicalVolumeStore.

  G4double envSizeXY = 0;
  G4double envSizeZ = 0;

  if (!fEnvelopeBox)
  {
    G4LogicalVolume* envLV
      = G4LogicalVolumeStore::GetInstance()->GetVolume("Envelope");
    if ( envLV ) fEnvelopeBox = dynamic_cast<G4Box*>(envLV->GetSolid());
  }

  if ( fEnvelopeBox ) {
    envSizeXY = fEnvelopeBox->GetXHalfLength()*2.;
    envSizeZ = fEnvelopeBox->GetZHalfLength()*2.;
  }
  else  {
    G4ExceptionDescription msg;
    msg << "Envelope volume of box shape not found.\n";
    msg << "Perhaps you have changed geometry.\n";
    msg << "The gun will be place at the center.";
    G4Exception("PrimaryGeneratorAction::GeneratePrimaries()",
     "MyCode0002",JustWarning,msg);
  }


 

if (true){
FILE *fp;
    fp = fopen("datos.txt","a+"); 
fprintf(fp, "%f %f %f\n", x0, y0 ,z0);
fclose(fp);



//FILE *fp;
 FILE   *fp2 = fopen("datos2.txt","a+"); 
fprintf(fp2, "%f %f %f %f\n", x0,x[2], y0, px[2]);
fclose(fp2);
}
//G4cout << x0 << " " << y0 << " " << z0 << G4endl;

  //fParticleGun->GeneratePrimaryVertex(anEvent);

//std::stringstream strFileName;
//	strFileName << "../ElectronDistributions_" << TarMat << "_" << AperRadius1 << "mm_" << AperRadius2 << "mm/InitialElectronDistribution.csv";
	G4String EDist_filename = "DatosEdgardo.txt";



	std::ifstream file;
		file.open("../DatosEdgardo.txt");
std::string line;
		//getline(file, line);

		// G4cout << line << G4endl;
		//G4double _X = 0.0, _X2 = 0.0, _XP = 0.0, _XP2 = 0.0, _XXP = 0.0, _Y = 0.0, _Y2 = 0.0, _YP = 0.0, _YP2 = 0.0, _YYP = 0.0, _Z = 0.0, _Z2 = 0.0, _P = 0.0, _P2 = 0.0;
G4int cont=0;
 //G4int nofEvents = run->GetNumberOfEvent();
		while (getline(file, line)&&(cont<1000000)) {
			//G4cout << line << G4endl;
			//while (!file.eof()){
//G4cout<<line<<G4endl;
	 		//getline(file, line);
			std::vector <float> v = {};
			split_c(line, &v);

			//G4cout << v.size() << G4endl;
			//G4cout << v.at(0) <<G4endl;
			G4double x0_1, y0_1, z_pri, y_pri, x_pri, z0_1, px_1, py_1, pz_1;
			//G4double mass = particle->GetPDGMass();
G4double mass = 1.0;
			x0_1 = v.at(1);
			px_1 = v.at(2)*7;
			y0_1 = v.at(3);
			//py_1 = v.at(4)*v.at(6)*mass;
			py_1 = v.at(4)*7;
			z0_1 = v.at(4);
			pz_1 = v.at(5)/sqrt(v.at(1)*v.at(2) + v.at(2)*v.at(2) + 1)*mass;
			x_pri= v.at(2)*v.at(10);
			y_pri= v.at(4)*v.at(10);
			z_pri= v.at(7);
			pz_1=v.at(10)*1000;

			//G4cout << line << G4endl;
			
//G4ThreeVector pos(0,80*cm, 0);



G4ThreeVector pos(0, -1,0);

			//G4cout << "Pos: " << pos << G4endl << "Mom: "  << G4endl;
			// G4cout << mom.mag() << G4endl;
			// fprintf(electrondis, "%d,%f,%f,%f,%f,%f,%f\n", 100, pos[0], pos[1], pos[2], mom[0], mom[1], mom[2]);
			if(cont%1000==0)G4cout << "\r" << "Counter: " << cont;
			cont++;
			fParticleGun->SetParticlePosition(pos);
			
			//PARA EL HAZ IDEAL
  			//fParticleGun->SetParticleMomentumDirection(G4ThreeVector( 0,-1,0));
			//PARA EL HAZ IMPORTADO
			//fParticleGun->SetParticleMomentumDirection(G4ThreeVector( x_pri,-7,y_pri));
			fParticleGun->SetParticleMomentum(G4ThreeVector( x_pri,-pz_1,y_pri));
			//G4cout<<"AAA " << x0_1 << " " << y0_1 <<" " << cont <<G4endl;
			fParticleGun->GeneratePrimaryVertex(anEvent);
			//G4cout << x0_1 << "  "<<px_1 << "  " <<y0_1<< "  "<<py_1<<  G4endl;
			//G4cout << x0_1 << "  "<< px_1<<"  " <<y0_1<< "  "<< py_1<<G4endl;				
			
			FILE   *fp3_1 = fopen("Datos00.txt","a+"); 
			fprintf(fp3_1, "%f %f %f %f\n", x0_1,x_pri, y0_1,y_pri );
			fclose(fp3_1);

			// print(v);
			// cout << v << endl;
		}
		file.close();
		
  



}
*/

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{

	G4ThreeVector pos, dir;
	G4double  twopi=6.28;
	
G4double  sinTheta = G4RandGauss::shoot(0, 0.003);
G4double  cosTheta = std::sqrt(1 - sinTheta*sinTheta);
G4double  phi = twopi*G4UniformRand();

	//dir.set(sinTheta*std::cos(phi), sinTheta*std::sin(phi), cosTheta);
	dir.set(0,-1,0);

G4double  GunRadius=0.1;


	G4double  rho = G4UniformRand()*GunRadius;
	G4double  alpha = G4UniformRand()*twopi;

G4double  GunMeanEnergy=10;
G4double  GunStdEnergy=0.1;

	pos.setX(rho*std::sin(alpha));
	pos.setZ(rho*std::cos(alpha));
	pos.setY(50 *cm); // the primary electrons are generated 5 mm before the target
	G4double  ek=G4RandGauss::shoot(GunMeanEnergy, GunStdEnergy);
	//RandomParticles++;
	
	
 
	fParticleGun->SetParticlePosition(pos*mm);
	fParticleGun->SetParticleEnergy(ek*MeV);
	fParticleGun->SetParticleMomentumDirection(dir);
	fParticleGun->GeneratePrimaryVertex(anEvent);

	/*FILE   *fp_1 = fopen("emitancia","a+"); 
			fprintf(fp_1, "%f %f %f %f %f %f\n", pos[0],pos[1],pos[2],dir[0],dir[1],dir[2]);
			
			fclose(fp_1);
*/

//G4cout<< fParticleGun->GetParticleMomentum()<<" X  " <<fParticleGun->GetParticleEnergy ()  <<"PRUEBA1"<<G4endl;

}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}


