import numpy as np
from scipy.special import expn


class CollisionOperations:

    def calcProbCol(self, t, r, z):
        """
        Calculates the probability of a collision happening in the time interval. Returns effective probability.
        """

        delT = self.delta_t  # model.feps*np.min(1/model.effColRates) #### change this to match global timestep

        self.PCol = 1 - np.exp(-delT * np.sum(self.colRates))
        self.effPCol = 1 - np.exp(-delT * np.sum(self.effColRates))

        return self.effPCol


    def determineCollisionP(self, t, r, z):
        """
        The function which determnines whether a collision has happened during a timestep or not. Returns True if a collision
        happened; false if not.
        """

        pCol = self.calcProbCol(t, r, z)

        if np.random.rand() <= pCol:

            collision = True
        else:
            collision = False

        self.seedNo += 1
        np.random.seed(self.seedNo)

        return collision


    def determineFragmentMass(self, mMax):
        """
        Auxiliary function in determining collision outcome. If fragmentation/erosion occurs, the
        monomer may end up in one of the fragments. This function determines the fragment mass in
        which the monomer is embedded if this occurs.

        mMax - Maximum fragment mass allowed by dynamics.

        See also Birnstiel et al. 2010; 2011
        """

        x = np.random.power(6 - 3 * self.para["x_frag"] + 1, size=1)[
            0]  # See Birnstiel+ 2011; we draw a value between 0 and 1, which is the range of possible values.
        self.seedNo += 1
        np.random.seed(self.seedNo)

        sMax = (3 * mMax / (4 * np.pi * self.monomer.homeAggregate.prop["rhoAgg"])) ** (1 / 3)

        size = x * (sMax - self.monomer.prop["sMon"]) + self.monomer.prop["sMon"]

        mass = 4 / 3 * np.pi * self.monomer.homeAggregate.prop["rhoAgg"] * size ** 3

        return mass


    def determineCollisionPartner(self, t, r, z):
        """
        Determines the size and mass of the collision partner.
        """

        colSum = np.sum(self.effColRates)

        colNum = colSum * np.random.rand()

        self.seedNo += 1
        np.random.seed(self.seedNo)

        compSum = 0
        ind = 0

        while (compSum < colNum) and (compSum < colSum):
            compSum += self.effColRates[ind]
            ind += 1

        if ind >= 100:
            size = self.grainSizes[-1]
            self.monomer.homeAggregate.prop["NCol"] = 1 / self.effectiveFact[-1]
        else:
            size = self.grainSizes[ind]
            self.monomer.homeAggregate.prop["NCol"] = 1 / self.effectiveFact[ind]

        # determines cloud size
        # print("Cloud size of collision partner: "+str(self.monomer.homeAggregate.prop["NCol"]))
        self.monomer.homeAggregate.prop["sCol"] = size
        self.monomer.homeAggregate.prop["mCol"] = 4 / 3 * np.pi * self.monomer.homeAggregate.prop["sCol"] ** 3 * \
                                                  self.monomer.homeAggregate.prop["rhoAgg"]

        if (self.debug) and (self.monomer.homeAggregate.prop["sAgg"] > self.grainSizes[-1]):
            print("Collision rates: ", self.effColRates, "/s")
            print("Determined collision partner size: ", self.monomer.homeAggregate.prop["sCol"])


    def determineCollisionOutcome(self, t, r, z, size=None):
        """
        Determines whether erosion, fragmentation or coagulation takes place. All input is in SI.
        """

        vRel = self.calcTotVRel(t, r, z, doPrint=False)  ### Note that calcTotVrel takes input in SI
        self.vRel = vRel
        # print(vRel)
        cond = 99
        # Determine whether fragmentation occurs
        if vRel >= self.para["v_frag"]:
            cond = 1
            fragmentation = True
        elif vRel < (self.para["v_frag"] - self.para["del_v_frag"]):
            cond = 2
            fragmentation = False
        else:
            cond = 3
            pFrag = 1 - (self.para["v_frag"] - vRel) / self.para["del_v_frag"]
            if np.random.rand() <= pFrag:

                fragmentation = True
            else:
                fragmentation = False

            self.seedNo += 1
            np.random.seed(self.seedNo)

        # if (vRel>self.para["v_frag"]):
        #     print(t/(self.sTOyr*1e3), vRel, fragmentation, cond)
        if fragmentation:  # if fragmentation we need to pick a new, smaller size of our home aggregate

            # does catastrophic disruption or erosion occur?
            massRat = self.monomer.homeAggregate.prop["mCol"] / self.monomer.homeAggregate.prop["mAgg"]
            mCol = self.monomer.homeAggregate.prop["mCol"]
            mAgg = self.monomer.homeAggregate.prop["mAgg"]

            mMin = self.monomer.prop["mMon"]

            if massRat <= self.mrat:
                # In this case the home aggregate is being eroded.

                mTot = 2 * mCol * self.monomer.homeAggregate.prop["NCol"]  # multiply with cloud size,

                # Is the monomer in the excavated mass? Two masses worth of mCol are excavated from
                # the home aggregate. Assuming random location,
                # the ejection probability is given by:

                pNotEj = (1 - mTot / mAgg) ** self.monomer.homeAggregate.prop["NCol"]  # probability of not
                # being ejected
                pEj = 1 - pNotEj
                # print("Ejection probability", pEj)

                if np.random.rand() <= pEj:  # the monomer is ejected and we need to determine the size of
                    # the fragment.

                    mMax = mCol
                    self.monomer.homeAggregate.prop["mAgg"] = self.determineFragmentMass(mMax)
                    message = "home aggregate eroded, monomer ejected"
                    outcome = "ejection"
                    interaction_id = 4

                else:  # some mass is eroded away from the home aggregate, but the monomer remains in
                    # the home aggregate.
                    self.monomer.homeAggregate.prop["mAgg"] -= mTot / 2
                    message = "home aggregate eroded by {:.1e} particles, monomer remained".format(
                        self.monomer.homeAggregate.prop["NCol"])
                    outcome = "erosion"
                    interaction_id = 3

                self.seedNo += 1
                np.random.seed(self.seedNo)

            elif massRat >= 1 / self.mrat:
                # In this case the home aggregate is the eroding particle.
                # Note that in this case, the collision partner is never a cloud.

                mTot = 2 * mAgg

                # Is the monomer in the excavated mass? ---> Assume no? May read a bit better into this.
                # - Brauer, Dullemond & Henning (2008)
                # - Hasegawa et al. (2021) - Simulations, v_frag is function of mass ratio.
                # For now, lets assume bullets: 2x mAgg of the collision partner are excavated such that the collision partner is the new home aggregate.
                # the impactor burries itself deep enough such that the excavated mass is solely originating from the collision partner.
                self.monomer.homeAggregate.prop["mAgg"] = self.monomer.homeAggregate.prop["mCol"] - mTot / 2
                message = "home aggregate impacted"
                outcome = "impact"
                interaction_id = 5


            else:
                # In all the other cases the home aggregate is catastrophically disrupted.

                self.monomer.homeAggregate.prop["mAgg"] = self.determineFragmentMass(max([mAgg, mCol]))

                # In which fragment is the monomer? What is the size of this fragment?

                message = "catastrophic disruption"
                outcome = "fragmentation"
                interaction_id = 2

            # to include: determine the new depth of the monomer and whether it is exposed

        else:  # otherwise we update it. Coagulation takes place.
            self.monomer.homeAggregate.prop["mAgg"] += self.monomer.homeAggregate.prop["mCol"] * \
                                                       self.monomer.homeAggregate.prop["NCol"]
            message = "coagulation with {:.1e} particle(s) of size {:.1e} m".format(self.monomer.homeAggregate.prop["NCol"],
                                                                                    self.monomer.homeAggregate.prop["sCol"])
            outcome = "coagulation"
            interaction_id = 1

        # In any case the new home aggregate size is calculated via the new mass.

        self.monomer.homeAggregate.prop["sAgg"] = (3 * self.monomer.homeAggregate.prop["mAgg"] / (
                    4 * np.pi * self.monomer.homeAggregate.prop["rhoAgg"])) ** (1 / 3)

        if self.monomer.homeAggregate.prop["sAgg"] < self.monomer.prop["sMon"]:
            # If the fragment is smaller than the monomer size there will be a free monomer.
            self.monomer.homeAggregate.prop["sAgg"] = self.monomer.prop["sMon"]
            self.monomer.homeAggregate.prop["mAgg"] = self.monomer.prop["mMon"]

        if (self.debug) and (self.monomer.homeAggregate.prop["sAgg"] > self.grainSizes[-1]):
            print("Home aggregate exceeded maximum mass in background distribution distribution")
            print("Event: ", outcome)
            print("Partner: ", self.monomer.homeAggregate.prop["sCol"], "m")
            print("Group size: ", self.monomer.homeAggregate.prop["NCol"])
            print("v_rel: ", vRel, "m/s")
            print(t, r, z)

        # Determine new depth of the monomer, and (if needed) if we are still exposed to the gas phase.
        if self.trackice:
            self.determineMonomerExposed(outcome)

        return message, interaction_id


    def doCollisions(self, r_in, z_in, t_in):
        r, z, t = self.unpackVars(r_in, z_in, t_in)

        collision = self.determineCollisionP(t, r, z)

        if collision:
            self.determineCollisionPartner(t, r, z)
            log, interaction_id = self.determineCollisionOutcome(t, r, z)
            self.sizeChanged = True
        else:
            log = "no collision happened"
            interaction_id = 0
            self.sizeChanged = False

        if t == 0:
            (self.monomer.sec_sol["interaction"]).append(interaction_id)
            (self.monomer.sec_sol["exposed"]).append(int(self.monomer.exposed))
        else:
            (self.monomer.sec_sol["interaction"]).append(interaction_id)
            (self.monomer.sec_sol["exposed"]).append(int(self.monomer.exposed))


    def determinePExp(self):
        tau = 3 / 4 * (self.monomer.prop["zMon"] - self.monomer.prop["zCrit"]) / self.monomer.prop["sMon"] * (
        self.monomer.homeAggregate.prop["phi"])
        pVals = expn(2, tau)  # We solve the exponential integral
        return pVals


    def determineRandomPos(self):
        depth = self.monomer.homeAggregate.prop["sAgg"] * (1 - np.random.rand() ** (1 / 3))
        self.seedNo += 1
        np.random.seed(self.seedNo)

        return depth


    def determineMonomerExposed(self, outcome, redoAgg=False):
        """
        Determines the depth in the new home aggregate at which the monomer is located. Method depends on collision outcome. Requires the new aggregate size
        to be known.

        Note to self: We still have the possibility to make the exposure probability vary as a function of distance from the home aggregate core. For now it is
        0 if z>z_crit or 1 if z<z_crit
        """

        # First we determine the new depth
        if outcome in ["coagulation", "erosion"]:
            sOld = self.monomer.sAgg_sol[-1]
            sNew = self.monomer.homeAggregate.prop["sAgg"]
            self.monomer.prop["zMon"] += sNew - sOld
            if self.monomer.prop["zMon"] < 0:
                self.monomer.prop["zMon"] = 0  # if erosion of the home aggregate leads to a negative z, we set it zero.
        elif outcome in ["fragmentation", "ejection", "impact"]:
            self.monomer.prop["zMon"] = self.determineRandomPos()

            # Subsequently whether the monomer is exposed at that depth
        if self.monomer.prop["zMon"] <= self.monomer.prop["zCrit"]:
            self.monomer.exposed = True
        else:
            pExp = self.determinePExp()
            if np.random.rand() <= pExp:
                self.monomer.exposed = True
            else:
                self.monomer.exposed = False

            self.seedNo += 1
            np.random.seed(self.seedNo)