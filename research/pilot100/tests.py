import unittest
from build import state_key, validate_position, eligible
from endgame.solver import Position
from endgame.fast import hand_from_text as h, move_from_text as m
class ContractTests(unittest.TestCase):
 def test_cross_hand_card_limit(self):
  with self.assertRaises(ValueError): validate_position(Position((h("333"),h("33"),h("4"))))
 def test_joker_limit(self):
  with self.assertRaises(ValueError): validate_position(Position((h("D"),h("D"),h("4"))))
 def test_equivalent_rank_shift_dedup(self):
  self.assertEqual(state_key(Position((h("334"),h("55"),h("66")))),state_key(Position((h("667"),h("88"),h("99")))))
 def test_turn_is_part_of_key(self):
  self.assertNotEqual(state_key(Position((h("3"),h("4"),h("5")),0)),state_key(Position((h("3"),h("4"),h("5")),1)))
 def test_unknown_and_losing_rejected(self):
  self.assertFalse(eligible({"status":"timeout","winning_moves":["3"]}))
  self.assertFalse(eligible({"status":"verified","winning_moves":[]}))
  self.assertTrue(eligible({"status":"verified","winning_moves":["3"]}))
 def test_invalid_pass_state(self):
  with self.assertRaises(ValueError): validate_position(Position((h("3"),h("4"),h("5")),0,None,1))


class SourceParserTests(unittest.TestCase):
 def make_row(self,last,actions):
  text=("Assume you are the landlord. The current cards for the landlord are: [3, 17]. "
  "The current cards for the peasant that plays before landlord are: [4]. "
  "The current cards for the peasant that plays after landlord are: [5]. "
  "The last two moves by other players are "+str(last)+". "
  "The legal actions for the landlord are: "+str(actions)+".")
  return {"row":{"sentence":text}}
 def test_newest_first_without_pass(self):
  from mastermind_extract import parse_row
  p=parse_row(self.make_row([[6],[]],[[17],[]]))
  self.assertEqual(p.passes,0); self.assertEqual(p.last.cards,(6,))
  self.assertEqual(p.hands[1],h("5"))
 def test_newest_pass_preserves_previous(self):
  from mastermind_extract import parse_row
  p=parse_row(self.make_row([[],[6]],[[17],[]]))
  self.assertEqual(p.passes,1); self.assertEqual(p.last.cards,(6,))
 def test_two_passes_reopen_lead(self):
  from mastermind_extract import parse_row
  p=parse_row(self.make_row([[],[]],[[3],[17]]))
  self.assertIsNone(p.last); self.assertEqual(p.passes,0)
 def test_inconsistent_legal_actions_reject(self):
  from mastermind_extract import parse_row
  with self.assertRaises(ValueError):parse_row(self.make_row([[6],[]],[[3],[17],[]]))
 def test_reference_team_and_pass_states(self):
  from reference_ordered import move_values
  from build import signature,Solver
  for p in [Position((h("77"),h("336"),h("QQ")),1),
            Position((h("34"),h("25"),h("69")),1,m("7"),1)]:
   want=[((t,c),w) for t,c,w in signature(Solver().move_values(p))]
   self.assertEqual(move_values(p),want)

if __name__=="__main__": unittest.main()
