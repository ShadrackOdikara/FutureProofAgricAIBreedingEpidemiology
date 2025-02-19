from src import merged_df, potato_disease_data, combined_predictions, epi_risks, city_info, matched_lines, cities_data, disease_data, response, app, determinant,score,sim_manager,r,determinant, top_match_score

#DataLoader()

print(merged_df)
print(combined_predictions)
print(potato_disease_data)
print(epi_risks)
print(city_info)
print(matched_lines)
#determinant = float(matched_lines[0].split("Risk Determinant: ")[1])
determinant = float(matched_lines[0].split("Risk Determinant: ")[1]) if matched_lines else 1
print(determinant)
print(score)
print(response.message.content)
#print(best_match)
print(run(scoring_input))
#print(sim_manager.simulate_future_steps())
#print("future of agrics")

